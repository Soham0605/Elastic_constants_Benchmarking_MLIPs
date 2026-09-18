"""
elastic_utils.py

Shared machinery for computing an elastic tensor with any ASE calculator,
using the same finite-strain philosophy as VASP's IBRION=6 / ISIF=3:
apply a small set of independent lattice strains, relax the ionic
positions at each fixed strained cell, fit stress vs. strain.

This module is calculator-agnostic -- compute_elastic_mlip.py supplies the
calculator, this file just does the strain/relax/fit/derive bookkeeping so
it's identical for every potential and directly comparable to the DFT
parsing in parse_dft_elastic.py.

Units: everything in here is GPa (stresses, Cij, K, G, E) unless stated.

*** VALIDATE BEFORE TRUSTING, ON A NEW CLUSTER/ENV ***
ASE and VASP do not use the same stress sign convention. This module's
strain-fitting path stays entirely within ASE's own convention (every
stress it sees comes from calc.get_stress()), so no sign correction is
needed here -- but parse_dft_elastic.py, which reads VASP's own output,
does need one (see that file's docstring). As a sanity check on a new
setup, run compute_elastic_mlip.py on a well-known reference (bulk Cu:
C11~168, C12~121, C44~75 GPa) and confirm you recover published values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.optimize import FIRE
from pymatgen.analysis.elasticity.elastic import ElasticTensor
from pymatgen.analysis.elasticity.strain import DeformedStructureSet, Strain
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.io.ase import AseAtomsAdaptor

EV_A3_TO_GPA = 160.21766208  # 1 eV/Angstrom^3 = 160.21766208 GPa

# Normal strains small, shear strains larger -- standard practice (matches
# e.g. atomate's elastic workflow) so the least-squares fit per Cij isn't
# just 2 points.
DEFAULT_NORM_STRAINS = (-0.01, -0.005, 0.005, 0.01)
DEFAULT_SHEAR_STRAINS = (-0.06, -0.03, 0.03, 0.06)


def relax_atoms_fixed_cell(atoms, calc, fmax=0.01, steps=200, logfile=None):
    """Relax ionic positions only (cell stays exactly as given) and return
    the relaxed Atoms plus the stress tensor (3x3, GPa) at that geometry."""
    atoms.calc = calc
    if len(atoms) > 1:
        opt = FIRE(atoms, logfile=logfile)
        opt.run(fmax=fmax, steps=steps)
    stress_ev_a3 = atoms.get_stress(voigt=False)  # 3x3, eV/Angstrom^3
    stress_gpa = stress_ev_a3 * EV_A3_TO_GPA
    return atoms, stress_gpa


def compute_elastic_tensor(
    structure,
    calc,
    norm_strains=DEFAULT_NORM_STRAINS,
    shear_strains=DEFAULT_SHEAR_STRAINS,
    fmax=0.01,
    verbose=True,
):
    """Given a relaxed pymatgen Structure and an ASE calculator, generate
    the deformed structure set, relax + get stress at each, and fit the
    elastic tensor. Returns (ElasticTensor, eq_stress)."""
    adaptor = AseAtomsAdaptor()

    eq_atoms = adaptor.get_atoms(structure)
    _, eq_stress_gpa = relax_atoms_fixed_cell(eq_atoms, calc, fmax=fmax)
    eq_stress = Stress(eq_stress_gpa)
    if verbose:
        print(f"  eq stress trace/3 (should be ~0 GPa if well relaxed): "
              f"{np.trace(eq_stress_gpa)/3:.4f}")

    dss = DeformedStructureSet(
        structure, norm_strains=norm_strains, shear_strains=shear_strains
    )

    strains, stresses = [], []
    for i, (deformation, deformed_structure) in enumerate(zip(dss.deformations, dss)):
        atoms = adaptor.get_atoms(deformed_structure)
        _, stress_gpa = relax_atoms_fixed_cell(atoms, calc, fmax=fmax)
        strain = Strain.from_deformation(deformation)
        strains.append(strain)
        stresses.append(Stress(stress_gpa))
        if verbose:
            mag = np.max(np.abs(strain))
            print(f"  [{i+1}/{len(dss)}] strain mag {mag:.3f} -> "
                  f"stress trace/3 {np.trace(stress_gpa)/3:.3f} GPa")

    et = ElasticTensor.from_independent_strains(
        strains, stresses, eq_stress=eq_stress, vasp=False
    )
    et = et.zeroed(tol=0.1).voigt_symmetrized

    return et, eq_stress


def chen_hardness(k_gpa: float, g_gpa: float) -> float:
    """Vickers hardness estimate, Chen et al. 2011 (Intermetallics 19:1275):
    Hv = 2*(k^2*G)^0.585 - 3, with k = G/K (Pugh's ratio), all in GPa."""
    k = g_gpa / k_gpa
    return 2.0 * (k**2 * g_gpa) ** 0.585 - 3.0


def derived_properties(et: ElasticTensor) -> dict:
    """Voigt-Reuss-Hill bulk/shear modulus and the six properties used for
    the spider plot: K, G, E, Poisson's ratio, B/G (Pugh), and Hv."""
    k_vrh = float(et.k_vrh)
    g_vrh = float(et.g_vrh)
    e_vrh = 9.0 * k_vrh * g_vrh / (3.0 * k_vrh + g_vrh)
    poisson = (3.0 * k_vrh - 2.0 * g_vrh) / (2.0 * (3.0 * k_vrh + g_vrh))
    b_over_g = k_vrh / g_vrh
    hv = chen_hardness(k_vrh, g_vrh)
    return {
        "K_vrh_GPa": k_vrh,
        "G_vrh_GPa": g_vrh,
        "E_vrh_GPa": e_vrh,
        "poisson_ratio": poisson,
        "B_over_G": b_over_g,
        "Hv_GPa": hv,
    }


def save_results(path: Path, potential: str, et: ElasticTensor, extra: dict | None = None):
    payload = {
        "potential": potential,
        "Cij_GPa": et.voigt.tolist(),
        **derived_properties(et),
    }
    if extra:
        payload.update(extra)
    Path(path).write_text(json.dumps(payload, indent=2))
    print(f"  wrote {path}")
    return payload

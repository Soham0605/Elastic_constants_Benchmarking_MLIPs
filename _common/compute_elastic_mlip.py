#!/usr/bin/env python3
"""
compute_elastic_mlip.py

Compute the elastic tensor of a relaxed structure with ONE MLIP, using the
same finite-strain approach as VASP IBRION=6/ISIF=3. Unlike the
equation-of-state benchmark, this needs no pre-generated set of deformed
structures on disk -- the strains are generated and evaluated entirely in
memory (see elastic_utils.compute_elastic_tensor).

Run from inside MLIPs/<potential>/, pointing --structure back at the
compound's relaxed structure. setup_directories.py generates a submit
script that does this for you -- you shouldn't normally need to invoke
this by hand.
"""

import argparse
import time
from pathlib import Path

from pymatgen.core import Structure

from mlip_calculators import get_calculator
from elastic_utils import compute_elastic_tensor, save_results

POTENTIALS = ["chgnet", "m3gnet", "mattersim", "orb_v3", "esen"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--potential", required=True, choices=POTENTIALS)
    p.add_argument("--structure", required=True,
                    help="Path to the relaxed structure (CONTCAR or any pymatgen-readable format)")
    p.add_argument("--outdir", default=".")
    p.add_argument("--fmax", type=float, default=0.01,
                    help="Force convergence (eV/Angstrom) for ionic relaxation at each strained cell")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    structure = Structure.from_file(args.structure)
    print(f"Loaded {structure.composition.reduced_formula} "
          f"({len(structure)} atoms) from {args.structure}")

    calc = get_calculator(args.potential)

    t0 = time.time()
    et, eq_stress = compute_elastic_tensor(structure, calc, fmax=args.fmax)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f} s")

    print("\nVoigt elastic tensor (GPa):")
    print(et.voigt)

    save_results(
        outdir / f"elastic_{args.potential}.json",
        args.potential,
        et,
        extra={"n_atoms": len(structure), "elapsed_s": elapsed,
               "source_structure": str(Path(args.structure).resolve())},
    )


if __name__ == "__main__":
    main()

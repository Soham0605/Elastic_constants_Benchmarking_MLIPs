"""
mlip_calculators.py

One factory function, `get_calculator(potential)`, that returns a ready-to-use
ASE Calculator for whichever MLIP you ask for.

IMPORTANT: this file is meant to live in ONE shared `_common/` directory and
be called from inside each potential's own conda env. Each branch below
only needs to import successfully in its *own* env -- you will never have
all 5 packages installed at once, and that's fine. If you get an
ImportError, it almost always means you're in the wrong `conda activate
<env>` for the potential you passed with --potential.

Authentication summary (verified, not guessed -- see README "MLIP requirements"):
  chgnet, m3gnet, mattersim, orb_v3  -- all auto-download from PUBLIC sources,
                                          no token/login needed.
  esen (UMA)                          -- REQUIRES a Hugging Face token (the
                                          repo is gated) plus a local checkpoint
                                          path. See config.yaml's mlip.esen section.
"""

import sys


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_calculator(potential: str):
    potential = potential.lower()
    device = "cuda" if _has_cuda() else "cpu"

    if potential == "chgnet":
        from chgnet.model import CHGNetCalculator
        return CHGNetCalculator(use_device=device)

    elif potential == "m3gnet":
        # matgl's pretrained models moved to the `materialyze` Hugging Face
        # org. IMPORTANT: the real repo IDs include the DFT functional in
        # the name (-PBE- or -r2SCAN-) -- a bare
        # "materialyze/TensorNet-PES-MatPES-2025.2" does not exist (HF
        # returns a 401 for both "private" and "doesn't exist" when
        # unauthenticated, which is what makes this mistake easy to make).
        # These are PUBLIC repos -- no token needed, just the correct name.
        # Defaults to PBE (matches a standard VASP PBE reference); set
        # M3GNET_MATPES_FUNCTIONAL=r2scan if your DFT uses r2SCAN instead.
        import os
        import matgl
        from matgl.ext.ase import PESCalculator
        import ase.units
        import numpy as np

        functional_key = os.environ.get("M3GNET_MATPES_FUNCTIONAL", "pbe").lower()
        functional_names = {"pbe": "PBE", "r2scan": "r2SCAN"}
        if functional_key not in functional_names:
            sys.exit(f"Unknown M3GNET_MATPES_FUNCTIONAL='{functional_key}', "
                      f"expected one of {list(functional_names)}")
        repo_id = f"materialyze/TensorNet-PES-MatPES-{functional_names[functional_key]}-2025.2"
        potential_model = matgl.load_model(repo_id)

        class StandardizedM3GNet(PESCalculator):
            def calculate(self, *args, **kwargs):
                # Run the standard MatGL calculation
                super().calculate(*args, **kwargs)
                
                # Convert the GPa stress output back to ASE standard eV/A^3
                if 'stress' in self.results and self.results['stress'] is not None:
                    self.results['stress'] = np.array(self.results['stress']) * ase.units.GPa

        return StandardizedM3GNet(potential=potential_model)

    elif potential == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device=device)

    elif potential == "orb_v3":
        # orb-models has moved the ORBCalculator class around across
        # versions without much notice -- as of this writing it actually
        # lives in orb_models.forcefield.inference.calculator, found by
        # grepping the installed package directly (`grep -rl "class
        # ORBCalculator" $(python -c "import orb_models,os;
        # print(os.path.dirname(orb_models.__file__))")`), NOT by reading
        # any changelog. Older/newer installs may differ, so this tries
        # every location seen in the wild rather than hardcoding one, and
        # fails with a self-diagnosis dump if none of them match.
        import importlib
        from orb_models.forcefield import pretrained

        candidates = [
            ("orb_models.forcefield.pretrained", "ORBCalculator"),
            ("orb_models.forcefield.calculator", "ORBCalculator"),
            ("orb_models.pretrained", "ORBCalculator"),
            ("orb_models.forcefield.atomic_system", "ORBCalculator"),
            ("orb_models.forcefield.inference.calculator", "ORBCalculator"),
        ]
        ORBCalculator = None
        errors = []
        for module_path, class_name in candidates:
            try:
                mod = importlib.import_module(module_path)
                ORBCalculator = getattr(mod, class_name)
                break
            except (ImportError, AttributeError) as e:
                errors.append(f"  {module_path}.{class_name}: {e}")

        if ORBCalculator is None:
            import pkgutil
            orb_models = importlib.import_module("orb_models")
            submodules = sorted(
                m.name for m in pkgutil.walk_packages(orb_models.__path__, prefix="orb_models.")
            )
            sys.exit(
                "Could not find ORBCalculator in any known location for your "
                "installed orb-models version. Tried:\n"
                + "\n".join(errors)
                + "\n\nAvailable orb_models submodules on this install:\n  "
                + "\n  ".join(submodules)
                + "\n\nFind the module containing 'class ORBCalculator' with e.g.:\n"
                  '  grep -rl "class ORBCalculator" $(python -c '
                  '"import orb_models, os; print(os.path.dirname(orb_models.__file__))")\n'
                  "and add its exact path as a new tuple in the orb_v3 branch of "
                  "mlip_calculators.py."
            )

        orbff = pretrained.orb_v3_conservative_inf_omat(device=device)
        # The newest orb-models API returns a tuple of (model, atoms_adapter)
        if isinstance(orbff, tuple) and len(orbff) == 2:
            return ORBCalculator(orbff[0], atoms_adapter=orbff[1], device=device)
        else:
            # Fallback for the older API
            return ORBCalculator(orbff, device=device)

    elif potential == "esen":
        # UMA checkpoints are gated on Hugging Face -- fairchem checks the
        # license acceptance even when loading from a local .pt path, so
        # HF_TOKEN must be set regardless of ESEN_CHECKPOINT being local.
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        import os

        if not os.environ.get("HF_TOKEN"):
            sys.exit(
                "HF_TOKEN is not set. This should be set by your submit script "
                "(via config.yaml's mlip.esen.hf_token_file) -- if you're "
                "seeing this, check that config.yaml has that field filled in."
            )

        ckpt = os.environ.get("ESEN_CHECKPOINT")
        if not ckpt:
            sys.exit(
                "ESEN_CHECKPOINT is not set. This should be set by your submit "
                "script (via config.yaml's mlip.esen.checkpoint) -- if you're "
                "seeing this, check that config.yaml has that field filled in."
            )
        predictor = load_predict_unit(path=ckpt, device=device)
        return FAIRChemCalculator(predictor, task_name="omat")

    else:
        raise ValueError(
            f"Unknown potential '{potential}'. "
            "Expected one of: chgnet, m3gnet, mattersim, orb_v3, esen."
        )
# elastic-bench

Benchmark machine-learned interatomic potentials (MLIPs) against DFT for
single-crystal elastic constants, on your own cluster, with your own VASP
license.

Given a relaxed ground-state structure, this tool:
1. Sets up and submits a DFT (VASP, `IBRION=6`/`ISIF=3`) elastic-tensor
   calculation
2. Runs the same structure through 5 MLIPs (CHGNet, M3GNet, MatterSim,
   Orb-v3, eSEN/UMA), each computing its own elastic tensor via finite
   strains
3. Derives the standard mechanical-property set from both -- bulk modulus
   K, shear modulus G, Young's modulus E, Poisson's ratio, the Pugh ratio
   B/G, and an estimated Vickers hardness Hv -- and plots them together as
   a spider (radar) chart with quantitative error metrics

This is the companion to `eos-bench` (same design, same config system,
same MLIP calculator factory) -- built second so any fix that turned out
to be general (the orb_v3 import path, the m3gnet Hugging Face repo name)
carries over directly.

---

## Requirements

Same as `eos-bench`:

1. A working VASP install with a valid license and an executable you can
   run on your cluster.
2. A SLURM cluster (the only scheduler supported right now).
3. Conda environments for each MLIP you want to use, each with that MLIP's
   own package installed. `--potentials` lets you run a subset.
4. Python 3.10+ with `ase`, `pymatgen`, and `pyyaml` on whatever node runs
   `setup_directories.py` and `configure.py`.

### MLIP requirements

| Potential | Auth needed? | Notes |
|---|---|---|
| CHGNet | No | Weights bundled with the package |
| M3GNet | No | Auto-downloads from a public Hugging Face org |
| MatterSim | No | Auto-downloads via the official `mattersim` package |
| Orb-v3 | No | Auto-downloads from a public URL |
| eSEN/UMA | Yes | Gated on Hugging Face -- needs a token + a local checkpoint |

Only eSEN/UMA needs anything beyond `pip install`. To use it:
1. Request access to the UMA model on Hugging Face at [https://huggingface.co/facebook/OMAT24](https://huggingface.co/facebook/OMAT24) (you must register and agree to their terms) and generate a User Access Token (**Settings -> Access Tokens**).
2. Save your token to a secured file: `echo "hf_yourTokenHere" > ~/.hf_token && chmod 600 ~/.hf_token` -- never put the token value directly in any script or config file, only in this private, permission-locked file.
3. Download the checkpoint `.pt` file you want (e.g. via `huggingface-cli`) and note its path.
`configure.py` (below) will ask for both paths and wire them in. If you don't have eSEN set up, just leave it out of `--potentials` -- everything else works independently.

### A note on orb_v3

`orb-models` has moved the `ORBCalculator` class between versions more
than once, without much notice. As of this writing it lives in
`orb_models.forcefield.inference.calculator` -- found by grepping the
installed package directly, not by reading a changelog:

```bash
grep -rl "class ORBCalculator" $(python -c "import orb_models, os; print(os.path.dirname(orb_models.__file__))")

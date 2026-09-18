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

Only eSEN/UMA needs anything beyond `pip install` -- see `eos-bench`'s
README for the exact steps (token generation, `~/.hf_token`, checkpoint
download); it's identical here.

### A note on orb_v3

`orb-models` has moved the `ORBCalculator` class between versions more
than once, without much notice. As of this writing it lives in
`orb_models.forcefield.inference.calculator` -- found by grepping the
installed package directly, not by reading a changelog:

```bash
grep -rl "class ORBCalculator" $(python -c "import orb_models, os; print(os.path.dirname(orb_models.__file__))")
```

`_common/mlip_calculators.py` tries every location seen in the wild (in
order) and falls back to printing every available submodule in your
install if none match, so you can find the right one yourself if this
moves again.

---

## Installation

```bash
git clone <this-repo-url> elastic-bench
cd elastic-bench
pip install ase pymatgen pyyaml --break-system-packages
```

## One-time setup

```bash
cp config.example.yaml config.yaml
python configure.py
```

Same wizard as `eos-bench` -- asks for your VASP path, module loads, SLURM
settings, conda env names, and (optionally) eSEN's token/checkpoint paths,
once, and validates them before you submit anything.

```bash
python configure.py --check     # re-validate later without re-prompting
```

---

## Workflow

Unlike `eos-bench`, there's no volume-series generation step here --
`IBRION=6`/`ISIF=3` computes the whole elastic tensor from a single VASP
calculation (VASP applies all the internal finite-difference strains
itself), and each MLIP does the equivalent strain sampling in memory, not
as separate directories on disk. So the actual workflow is shorter:

### 1. Prepare your compound directory

Before running anything here, you need a directory with your relaxed
structure and DFT inputs already in place:

```
Elastic_constants/YB2/
  CONTCAR         <- relaxed structure
  INCAR           <- must have IBRION=6, ISIF=3
  POTCAR
  KPOINTS
```

This tool doesn't generate any of these -- only the submit scripts around
them.

### 2. Build the submit scripts

```bash
python setup_directories.py \
    --base /path/to/Elastic_constants \
    --compounds YB2
```

Writes:
- `<compound>/submit_dft_elastic.sbatch` -- one VASP job
- `<compound>/MLIPs/<potential>/submit_<potential>.sbatch` for each
  potential in `--potentials` (default: all 5)
- `<compound>/submit_all_mlips.sh`

Every script pins its own working directory (`#SBATCH --chdir=...`), so it
behaves identically regardless of where you happen to run `sbatch` from.

### 3. Run DFT

```bash
sbatch /path/to/Elastic_constants/YB2/submit_dft_elastic.sbatch
```

### 4. Run the MLIPs

Independent of DFT -- only needs `CONTCAR`, so run any time after step 1:

```bash
bash /path/to/Elastic_constants/YB2/submit_all_mlips.sh
```

### 5. Parse the DFT results

Once step 3 finishes:

```bash
cd /path/to/Elastic_constants/YB2
python /path/to/elastic-bench/_common/parse_dft_elastic.py --outcar OUTCAR --outdir .
```

Writes `elastic_dft.json`, same schema the MLIP runs produce.

### 6. Plot

```bash
python /path/to/elastic-bench/plots/plot_spider.py \
    --base /path/to/Elastic_constants \
    --compounds YB2 \
    --outdir /path/to/Elastic_constants/spider_plots
```

One spider panel per compound: DFT traces a perfect hexagon at radius 1.0
(every property normalized to its DFT value), each MLIP's deviation from
that hexagon is immediately visible across all six axes -- K, G, E, Hv,
B/G, and Poisson's ratio -- despite them living on very different absolute
scales. A shaded band marks "within 15% of DFT" (`--agree-tol` to change
it). Below the panels, a color-coded table gives the mean absolute %error
per potential per compound. Pass multiple `--compounds` for a combined
figure.

---

## The physics, briefly

Every strain-fitting step stays entirely within ASE's own stress
convention (every stress value comes from `calc.get_stress()`), so no sign
correction is needed on the MLIP side. `parse_dft_elastic.py` reads
VASP's own `TOTAL ELASTIC MODULI` block directly, which VASP already
reports in the standard physical convention.

At each strained cell, only ionic positions are relaxed -- the strain
itself (imposed on the lattice vectors) is exactly what
`DeformedStructureSet` generates, matching VASP's own IBRION=6 approach of
finite-differencing stress with respect to a small set of independent
strains.

The elastic-tensor fitting pipeline (`elastic_utils.py`) is validated
against a real end-to-end test using ASE's bundled EMT calculator on bulk
copper: the result comes out exactly cubic-symmetric (all three diagonal
constants agree to 10+ decimal places) and lands close to real experimental
Cu values (C11~166 vs. 168 GPa published, C12~111 vs. 121, C44~95 vs. 75)
even though EMT is only an approximate potential -- strong evidence the
fitting machinery itself, independent of any specific MLIP's accuracy, is
correct.

Hardness (Hv) uses the Chen et al. 2011 empirical model
(`Hv = 2*(k^2*G)^0.585 - 3`, `k = G/K`) -- a widely-used but genuinely
empirical correlation, not a derived physical law; treat it as a useful
screening heuristic, not a measured hardness value.

---

## Directory structure this produces

```
Elastic_constants/
  YB2/
    CONTCAR, INCAR, POTCAR, KPOINTS
    OUTCAR, vasprun.xml, ...          <- after step 3
    elastic_dft.json                   <- after step 5
    submit_dft_elastic.sbatch
    submit_all_mlips.sh
    MLIPs/
      chgnet/    submit_chgnet.sbatch, elastic_chgnet.json (after step 4)
      m3gnet/    ...
      mattersim/ ...
      orb_v3/    ...
      esen/      ...
  spider_plots/
    All_Compounds_Elastic.png, .pdf
```

---

## Troubleshooting

A DFT or MLIP job dies almost instantly with no useful log content: check
`#SBATCH --chdir` in the generated script points where you expect -- every
script pins it explicitly, so this class of bug shouldn't recur, but if
you're invoking anything by hand, remember SLURM does not `cd` into the
script's own directory automatically.

`m3gnet` fails with a 401 / "Repository Not Found": the correct Hugging
Face repo ID includes the DFT functional in the name
(`materialyze/TensorNet-PES-MatPES-PBE-2025.2`) -- already fixed in
`_common/mlip_calculators.py`.

`orb_v3` fails with an `ORBCalculator` import error: see "A note on
orb_v3" above -- the grep command there finds the right module for your
specific installed version if it's moved again since this was written.

`esen` exits asking for `HF_TOKEN`/`ESEN_CHECKPOINT`: run
`python configure.py --check` to confirm both are set in `config.yaml`'s
`mlip.esen` section and that the files they point to actually exist.

---

## Limitations & roadmap

- SLURM only.
- No automatic queue monitoring -- you check job status and move to the
  next step yourself.
- Hv is a hardness estimate from an empirical model, not a DFT-computed
  quantity -- see "The physics, briefly" above.

---

## File reference

| File | Runs on | Purpose |
|---|---|---|
| `config.example.yaml` | -- | Template; copy to `config.yaml` and fill in |
| `config.py` | login node | Loads/validates `config.yaml` |
| `configure.py` | login node | Interactive setup wizard |
| `setup_directories.py` | login node | Writes all the sbatch scripts |
| `_common/mlip_calculators.py` | compute node (in each MLIP's env) | Returns an ASE calculator for a given potential |
| `_common/elastic_utils.py` | compute node | Shared strain/relax/fit/derived-property machinery |
| `_common/compute_elastic_mlip.py` | compute node (in each MLIP's env) | Runs one potential's elastic-tensor calculation |
| `_common/parse_dft_elastic.py` | login node (after DFT finishes) | Extracts the DFT elastic tensor into the same JSON schema |
| `plots/plot_spider.py` | login node (after everything finishes) | The final comparison figure |

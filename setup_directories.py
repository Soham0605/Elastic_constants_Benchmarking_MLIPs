#!/usr/bin/env python3
"""
setup_directories.py

Run this once per compound to get everything ready:
  1. (optionally) writes a DFT submit script -- a SINGLE VASP job (IBRION=6,
     ISIF=3 computes the full elastic tensor internally from one relaxed
     structure; unlike equation-of-state, there's no set of deformed
     structures to pre-generate here).
  2. Writes MLIPs/<potential>/submit_<potential>.sbatch for each requested
     potential, each calling the shared compute_elastic_mlip.py, which does
     its own strain sampling in memory.

Every cluster-specific detail (VASP path, module loads, conda init script,
SLURM account/qos/constraint, your username/email) comes from config.yaml.
Run `python configure.py` first if you haven't already.

Your compound directory needs, before running this:
  CONTCAR (or another relaxed structure), INCAR (with IBRION=6, ISIF=3),
  POTCAR, KPOINTS -- this tool does not generate any of those, only the
  submit scripts around them.

Usage:
    python setup_directories.py --base /path/to/Elastic_constants --compounds YB2 YB4
"""

import argparse
from pathlib import Path

from config import ConfigError, load_config

DFT_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={compound}_elastic
#SBATCH --chdir={compound_dir}
{account_line}{qos_line}{partition_line}{constraint_line}{mail_lines}#SBATCH --ntasks={ntasks}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --output=logs/elastic_%j.log
#SBATCH --error=logs/elastic_%j.err

module purge
{module_loads}
VASP_BIN="{vasp_executable}"

srun --mpi=pmix --cpu-bind=none "$VASP_BIN" > vasp.out 2>&1

echo "Elastic-constant DFT run for {compound} completed."
"""

MLIP_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={compound}_{potential}_elastic
#SBATCH --chdir={mlip_dir}
{account_line}{qos_line}{partition_line}{constraint_line}{mail_lines}#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --nodes=1

export OMP_NUM_THREADS={cpus_per_task}
export MKL_NUM_THREADS={cpus_per_task}

module purge
source {conda_init_script}
conda activate {env}
{secret_lines}
python {common_dir}/compute_elastic_mlip.py \\
    --potential {potential} \\
    --structure {structure} \\
    --outdir .
"""


def _sbatch_optional_line(flag: str, value):
    return f"#SBATCH --{flag}={value}\n" if value else ""


def _mail_lines(email):
    if not email:
        return ""
    return f"#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user={email}\n"


def write_dft_script(compound_dir: Path, compound: str, cfg: dict):
    (compound_dir / "logs").mkdir(parents=True, exist_ok=True)
    cluster = cfg["cluster"]
    modules = cfg["vasp"].get("module_loads") or []
    content = DFT_SBATCH_TEMPLATE.format(
        compound=compound,
        compound_dir=compound_dir.resolve(),
        account_line=_sbatch_optional_line("account", cluster.get("account")),
        qos_line=_sbatch_optional_line("qos", cluster.get("qos")),
        partition_line=_sbatch_optional_line("partition", cluster.get("partition")),
        constraint_line=_sbatch_optional_line("constraint", cluster.get("constraint")),
        mail_lines=_mail_lines(cfg["user"].get("email")),
        ntasks=cluster["dft"]["ntasks"],
        mem=cluster["dft"]["mem"],
        time=cluster["dft"]["time"],
        module_loads="\n".join(f"module load {m}" for m in modules),
        vasp_executable=cfg["vasp"]["executable"],
    )
    path = compound_dir / "submit_dft_elastic.sbatch"
    path.write_text(content)
    path.chmod(0o755)
    return path


def write_mlip_script(mlip_dir: Path, compound: str, potential: str, common_dir: Path,
                       structure: Path, cfg: dict):
    cluster = cfg["cluster"]
    secret_lines = []
    if potential == "esen":
        esen_cfg = cfg["mlip"].get("esen", {})
        token_file = esen_cfg.get("hf_token_file")
        checkpoint = esen_cfg.get("checkpoint")
        if not token_file or not checkpoint:
            raise ConfigError(
                "potential 'esen' requires mlip.esen.hf_token_file and "
                "mlip.esen.checkpoint to be set in config.yaml (run `python "
                "configure.py` to fill these in, or drop esen from --potentials)."
            )
        secret_lines.append(f"export HF_TOKEN=$(cat {token_file})")
        secret_lines.append(f"export ESEN_CHECKPOINT={checkpoint}")

    content = MLIP_SBATCH_TEMPLATE.format(
        compound=compound,
        potential=potential,
        mlip_dir=mlip_dir.resolve(),
        account_line=_sbatch_optional_line("account", cluster.get("account")),
        qos_line=_sbatch_optional_line("qos", cluster.get("qos")),
        partition_line=_sbatch_optional_line("partition", cluster.get("partition")),
        constraint_line=_sbatch_optional_line("constraint", cluster.get("constraint")),
        mail_lines=_mail_lines(cfg["user"].get("email")),
        ntasks=cluster["mlip"]["ntasks"],
        cpus_per_task=cluster["mlip"]["cpus_per_task"],
        mem=cluster["mlip"]["mem"],
        time=cluster["mlip"]["time"],
        conda_init_script=cfg["mlip"]["conda_init_script"],
        env=cfg["mlip"]["envs"][potential],
        secret_lines="\n".join(secret_lines),
        common_dir=common_dir,
        structure=structure,
    )
    (mlip_dir / "logs").mkdir(parents=True, exist_ok=True)
    path = mlip_dir / f"submit_{potential}.sbatch"
    path.write_text(content)
    path.chmod(0o755)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Directory containing your compound subdirectories")
    p.add_argument("--compounds", nargs="+", required=True)
    p.add_argument("--potentials", nargs="+",
                    default=["chgnet", "m3gnet", "mattersim", "orb_v3", "esen"])
    p.add_argument("--structure-name", default="CONTCAR",
                    help="Filename of the relaxed structure inside each compound dir")
    p.add_argument("--config", default=None, help="Path to config.yaml (default: ./config.yaml)")
    p.add_argument("--skip-dft-script", action="store_true",
                    help="Don't (re)write submit_dft_elastic.sbatch, e.g. if DFT is already done")
    args = p.parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        raise SystemExit(f"Config error: {e}")

    base = Path(args.base)
    common_dir = Path(__file__).resolve().parent / "_common"

    for compound in args.compounds:
        compound_dir = base / compound
        if not compound_dir.is_dir():
            print(f"!! {compound_dir} does not exist, skipping")
            continue

        structure_path = compound_dir / args.structure_name
        if not structure_path.exists():
            print(f"!! {structure_path} not found -- check --structure-name")

        if not args.skip_dft_script:
            dft_script = write_dft_script(compound_dir, compound, cfg)
            print(f"  wrote {dft_script}")

        mlips_dir = compound_dir / "MLIPs"
        mlip_scripts = []
        for potential in args.potentials:
            pot_dir = mlips_dir / potential
            pot_dir.mkdir(parents=True, exist_ok=True)
            rel_structure = Path("..") / ".." / args.structure_name  # MLIPs/<pot>/ -> compound_dir
            try:
                script = write_mlip_script(pot_dir, compound, potential, common_dir, rel_structure, cfg)
            except ConfigError as e:
                print(f"  !! skipping {potential}: {e}")
                continue
            mlip_scripts.append(script)
            print(f"  wrote {script}")

        submit_all = compound_dir / "submit_all_mlips.sh"
        lines = ["#!/bin/bash", "set -e"]
        for script in mlip_scripts:
            lines.append(f'(cd "{script.parent}" && sbatch "{script.name}")')
        submit_all.write_text("\n".join(lines) + "\n")
        submit_all.chmod(0o755)
        print(f"  wrote {submit_all}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
configure.py

Run this once to set up config.yaml -- prompts for your VASP path, conda
environments, cluster scheduler settings, then validates everything it can
before you submit a single job.

Usage:
    python configure.py            # interactive wizard, writes config.yaml
    python configure.py --check    # re-validate an existing config.yaml, no prompts
"""

import argparse
import sys
from pathlib import Path

import yaml

from config import check_environment, ConfigError, load_config, REQUIRED_POTENTIALS

CONFIG_PATH = Path("config.yaml")
EXAMPLE_PATH = Path("config.example.yaml")


def ask(prompt: str, default=None, required=True):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        val = input(f"{prompt}{suffix}: ").strip()
        if not val:
            if default is not None:
                return default
            if not required:
                return None
            print("  This field is required.")
            continue
        return val


def ask_yesno(prompt: str, default=True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    val = input(f"{prompt}{suffix}: ").strip().lower()
    if not val:
        return default
    return val.startswith("y")


def run_wizard():
    if not EXAMPLE_PATH.exists():
        sys.exit(f"{EXAMPLE_PATH} not found -- run this from the toolkit's root directory.")

    if CONFIG_PATH.exists():
        existing = yaml.safe_load(CONFIG_PATH.read_text())
        already_configured = bool((existing or {}).get("vasp", {}).get("executable"))
        if already_configured and not ask_yesno(
                f"{CONFIG_PATH} already has settings in it. Overwrite?", default=False):
            print("Leaving config.yaml untouched. Run `python configure.py --check` "
                  "to validate it instead.")
            return

    cfg = yaml.safe_load(EXAMPLE_PATH.read_text())

    print("\n--- Cluster / scheduler ---")
    cfg["cluster"]["account"] = ask("SLURM --account (blank if not used)", default="", required=False) or None
    cfg["cluster"]["qos"] = ask("SLURM --qos (blank if not used)", default="", required=False) or None
    cfg["cluster"]["partition"] = ask("SLURM --partition (blank if not used)", default="", required=False) or None
    cfg["cluster"]["constraint"] = ask("SLURM --constraint (blank if not used)", default="", required=False) or None

    print("\n--- User info ---")
    cfg["user"]["username"] = ask("Your cluster username")
    cfg["user"]["email"] = ask("Email for job notifications (blank to disable)", default="", required=False) or None

    print("\n--- VASP ---")
    cfg["vasp"]["executable"] = ask("Absolute path to your VASP executable (e.g. vasp_std)")
    modules_raw = ask("Modules to load before running VASP, comma-separated "
                       "(e.g. intel/2025.1.0,openmpi/5.0.7)", default="", required=False)
    cfg["vasp"]["module_loads"] = [m.strip() for m in modules_raw.split(",") if m.strip()]

    print("\n--- MLIPs ---")
    print("Path to your conda installation's conda.sh (e.g. ~/miniconda3/etc/profile.d/conda.sh)")
    cfg["mlip"]["conda_init_script"] = ask("conda.sh path")

    print("\nFor each potential, give the name of the conda env you installed it in.")
    print("(These do NOT need to match the potential's own name -- use whatever you actually named them.)")
    for pot in REQUIRED_POTENTIALS:
        cfg["mlip"]["envs"][pot] = ask(f"  conda env for {pot}", default=pot)

    print("\n--- eSEN/UMA (the only potential needing a secret -- see README) ---")
    if ask_yesno("Do you have eSEN/UMA set up?", default=False):
        cfg["mlip"]["esen"]["hf_token_file"] = ask(
            "Path to a private file containing just your Hugging Face token "
            "(will be chmod 600'd)", default="~/.hf_token")
        cfg["mlip"]["esen"]["checkpoint"] = ask("Absolute path to the UMA .pt checkpoint file")
    else:
        print("  Skipping -- you can fill in mlip.esen.* in config.yaml later, or drop "
              "'esen' from --potentials when running setup_directories.py.")

    CONFIG_PATH.write_text(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
    print(f"\nWrote {CONFIG_PATH.resolve()}")

    token_file = cfg["mlip"]["esen"].get("hf_token_file")
    if token_file:
        expanded = Path(token_file).expanduser()
        if expanded.exists():
            expanded.chmod(0o600)
            print(f"Set {expanded} to chmod 600 (private).")
        else:
            print(f"Note: {expanded} doesn't exist yet -- create it with your HF token "
                  f"before running eSEN, then `chmod 600` it yourself.")

    print("\nValidating...\n")
    check_environment(cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true",
                    help="Validate the existing config.yaml without prompting for anything")
    args = p.parse_args()

    if args.check:
        try:
            cfg = load_config()
        except ConfigError as e:
            sys.exit(f"Config error: {e}")
        ok = check_environment(cfg)
        sys.exit(0 if ok else 1)
    else:
        run_wizard()


if __name__ == "__main__":
    main()

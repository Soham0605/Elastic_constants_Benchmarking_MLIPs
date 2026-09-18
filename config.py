"""
config.py

Every script in this toolkit calls load_config() instead of hardcoding a
username, a VASP path, a module name, or a conda env name. This is the one
place cluster-specific information lives -- same design as the eos-bench
package this one is modeled on.

Usage in another script:
    from config import load_config
    cfg = load_config()          # looks for ./config.yaml by default
    vasp_bin = cfg["vasp"]["executable"]
"""

import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")

REQUIRED_POTENTIALS = ["chgnet", "m3gnet", "mattersim", "orb_v3", "esen"]


class ConfigError(Exception):
    pass


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(
            f"No config file found at {path.resolve()}.\n"
            f"Run this first:\n"
            f"  cp config.example.yaml config.yaml\n"
            f"  python configure.py\n"
            f"then fill in / confirm the values it asks about."
        )
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    _validate(cfg, path)
    return cfg


def _require(cfg: dict, dotted_path: str, path: Path):
    node = cfg
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"{path}: missing required field '{dotted_path}'")
        node = node[key]
    if node is None or node == "":
        raise ConfigError(
            f"{path}: '{dotted_path}' is not set. Run `python configure.py` "
            f"to fill it in, or edit config.yaml directly."
        )
    return node


def _validate(cfg: dict, path: Path):
    for field in ["vasp.executable", "user.username", "mlip.conda_init_script"]:
        _require(cfg, field, path)

    envs = cfg.get("mlip", {}).get("envs", {})
    missing = [p for p in REQUIRED_POTENTIALS if p not in envs or not envs[p]]
    if missing:
        raise ConfigError(
            f"{path}: mlip.envs is missing an entry for: {', '.join(missing)}"
        )

    scheduler = cfg.get("cluster", {}).get("scheduler")
    if scheduler != "slurm":
        raise ConfigError(
            f"{path}: cluster.scheduler = {scheduler!r}, but only 'slurm' is "
            f"supported right now (see README.md for the current limitation)."
        )


def check_environment(cfg: dict, verbose: bool = True) -> bool:
    """Best-effort validation that the paths/envs in config.yaml actually
    exist on THIS machine -- catches typos before you burn cluster time on
    a job that dies in the first second. Returns True if everything checked
    out; prints (and returns False on) problems otherwise."""
    import os
    import subprocess

    ok = True

    def report(passed, msg):
        nonlocal ok
        symbol = "OK  " if passed else "FAIL"
        if verbose:
            print(f"  [{symbol}] {msg}")
        if not passed:
            ok = False

    vasp_bin = cfg["vasp"]["executable"]
    report(os.path.isfile(vasp_bin) and os.access(vasp_bin, os.X_OK),
           f"VASP executable exists and is executable: {vasp_bin}")

    conda_init = cfg["mlip"]["conda_init_script"]
    report(os.path.isfile(conda_init), f"conda init script exists: {conda_init}")

    try:
        result = subprocess.run(["conda", "env", "list"], capture_output=True,
                                 text=True, timeout=30)
        existing_envs = {
            line.split()[0] for line in result.stdout.splitlines()
            if line and not line.startswith("#")
        }
        for potential, env_name in cfg["mlip"]["envs"].items():
            report(env_name in existing_envs,
                   f"conda env for {potential} exists: '{env_name}'")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        if verbose:
            print("  [SKIP] Could not run `conda env list` -- check env names manually")

    esen_cfg = cfg.get("mlip", {}).get("esen", {})
    hf_token_file = esen_cfg.get("hf_token_file")
    if hf_token_file:
        import os as _os
        expanded = _os.path.expanduser(hf_token_file)
        report(_os.path.isfile(expanded), f"HF token file exists: {hf_token_file}")
    else:
        if verbose:
            print("  [SKIP] mlip.esen.hf_token_file not set -- eSEN/UMA will not run without it")

    checkpoint = esen_cfg.get("checkpoint")
    if checkpoint:
        report(os.path.isfile(checkpoint), f"eSEN/UMA checkpoint exists: {checkpoint}")
    else:
        if verbose:
            print("  [SKIP] mlip.esen.checkpoint not set -- eSEN/UMA will not run without it")

    return ok


if __name__ == "__main__":
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded config.yaml OK. Checking environment...\n")
    passed = check_environment(cfg)
    sys.exit(0 if passed else 1)

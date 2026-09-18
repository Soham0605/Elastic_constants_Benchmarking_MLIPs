#!/usr/bin/env python3
"""
parse_dft_elastic.py

Pull the elastic tensor out of a completed VASP IBRION=6, ISIF=3 run
(OUTCAR) and save it in the same JSON format compute_elastic_mlip.py uses,
so plot_spider.py treats DFT as just another "potential".

Tries pymatgen's built-in Outcar elastic-tensor parsing first, and falls
back to a manual regex parse of the "TOTAL ELASTIC MODULI (kBar)" block if
that's unavailable in your pymatgen version.

Usage:
    python parse_dft_elastic.py --outcar OUTCAR --outdir .
    (run from inside the compound directory)
"""

import argparse
import re
from pathlib import Path

import numpy as np
from pymatgen.analysis.elasticity.elastic import ElasticTensor
from pymatgen.io.vasp.outputs import Outcar

from elastic_utils import save_results

KBAR_TO_GPA = 0.1


def _manual_parse(outcar_path: Path) -> np.ndarray:
    text = outcar_path.read_text()
    m = re.search(
        r"TOTAL ELASTIC MODULI \(kBar\)\s*\n"
        r"\s*Direction\s+.*\n"
        r"\s*-+\s*\n"
        r"((?:.*\n){6})",
        text,
    )
    if not m:
        raise RuntimeError(
            f"Could not find 'TOTAL ELASTIC MODULI (kBar)' block in {outcar_path}. "
            "Is this really a completed IBRION=6/ISIF=3 run?"
        )
    rows = m.group(1).strip().split("\n")
    cij_kbar = np.array(
        [[float(x) for x in row.split()[1:7]] for row in rows]
    )
    return cij_kbar * KBAR_TO_GPA


def get_dft_elastic_tensor(outcar_path: Path) -> ElasticTensor:
    try:
        outcar = Outcar(str(outcar_path))
        outcar.read_elastic_tensor()
        cij_gpa = np.array(outcar.data["elastic_tensor"]) * KBAR_TO_GPA
        print("  parsed elastic tensor via pymatgen Outcar")
    except Exception as e:  # noqa: BLE001 -- deliberately broad, this is a fallback path
        print(f"  pymatgen Outcar parsing failed ({e}); falling back to manual regex parse")
        cij_gpa = _manual_parse(outcar_path)

    return ElasticTensor.from_voigt(cij_gpa)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outcar", default="OUTCAR")
    p.add_argument("--outdir", default=".")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    et = get_dft_elastic_tensor(Path(args.outcar))
    print("\nDFT Voigt elastic tensor (GPa):")
    print(et.voigt)

    save_results(outdir / "elastic_dft.json", "dft", et,
                 extra={"source_outcar": str(Path(args.outcar).resolve())})


if __name__ == "__main__":
    main()

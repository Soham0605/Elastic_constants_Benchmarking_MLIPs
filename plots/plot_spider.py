#!/usr/bin/env python3
"""
plot_spider.py

Publication-style 1xN spider (radar) panel: one polar axis per compound,
each showing DFT (reference hexagon at 1.0) vs. every available MLIP, with
properties normalized to the DFT value -- DFT always traces a perfect
regular hexagon, and every MLIP's deviation from it is immediately visible,
even though K/G/E (tens-hundreds of GPa), Hv (single-digit-to-tens GPa),
B/G and Poisson's ratio (dimensionless, very different scale) live on
wildly different absolute scales.

Below the panels, a combined metrics table gives the mean absolute %error
per potential per compound (averaged over all 6 properties), plus an "All"
aggregate column.

Usage:
    python plots/plot_spider.py \
        --base Elastic_constants \
        --compounds YB2 YB4 B4C \
        --outdir Elastic_constants/spider_plots
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# Order requested: K, G, E, Hv, B/G, Poisson ratio
PROPERTIES = [
    ("K_vrh_GPa", "K\n(bulk)"),
    ("G_vrh_GPa", "G\n(shear)"),
    ("E_vrh_GPa", "E\n(Young's)"),
    ("Hv_GPa", "H$_v$\n(hardness)"),
    ("B_over_G", "B/G\n(Pugh)"),
    ("poisson_ratio", "Poisson's\nratio"),
]

POTENTIALS = ["chgnet", "m3gnet", "mattersim", "orb_v3", "esen"]
DISPLAY_NAMES = {
    "chgnet": "CHGNet", "m3gnet": "M3GNet", "mattersim": "MatterSim",
    "orb_v3": "Orb-v3", "esen": "eSEN/UMA", "DFT": "DFT (ref.)",
}
COLORS = {
    "DFT": "#000000", "chgnet": "#0072B2", "m3gnet": "#E69F00",
    "mattersim": "#009E73", "orb_v3": "#D55E00", "esen": "#CC79A7",
}
LINESTYLES = {"DFT": "-", "chgnet": "--", "m3gnet": "-.",
              "mattersim": (0, (3, 1, 1, 1)), "orb_v3": "--", "esen": ":"}
MARKERS = {"DFT": None, "chgnet": "o", "m3gnet": "s",
           "mattersim": "^", "orb_v3": "D", "esen": "v"}
PANEL_LABELS = list("abcdefgh")


def format_compound_name(name: str) -> str:
    return re.sub(r"(\d+)", r"$_{\1}$", name)


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_compound_data(base: Path, compound: str):
    compound_dir = base / compound
    dft = load_json(compound_dir / "elastic_dft.json")
    mlips = {}
    for pot in POTENTIALS:
        d = load_json(compound_dir / "MLIPs" / pot / f"elastic_{pot}.json")
        if d is not None:
            mlips[pot] = d
    return dft, mlips


def normalized_values(entry: dict, dft: dict):
    return [entry[key] / dft[key] for key, _ in PROPERTIES]


def compute_mape_table(base: Path, compounds: list):
    """Mean absolute %error vs. DFT, per potential per compound, averaged
    over the 6 properties: mean(|MLIP/DFT - 1|) * 100."""
    table = {pot: {} for pot in POTENTIALS}
    all_errors = {pot: [] for pot in POTENTIALS}

    for compound in compounds:
        dft, mlips = load_compound_data(base, compound)
        for pot in POTENTIALS:
            if dft is None or pot not in mlips:
                table[pot][compound] = None
                continue
            raw = normalized_values(mlips[pot], dft)
            errors = [abs(v - 1.0) * 100 for v in raw]
            table[pot][compound] = float(np.mean(errors))
            all_errors[pot].extend(errors)

    overall = {
        pot: (float(np.mean(errs)) if errs else None)
        for pot, errs in all_errors.items()
    }
    return table, overall


def plot_panel(ax, base: Path, compound: str, r_max: float, agree_tol: float,
                legend_handles: dict, panel_label: str):
    dft, mlips = load_compound_data(base, compound)

    n = len(PROPERTIES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _, label in PROPERTIES], fontsize=7)
    ax.tick_params(axis="x", pad=6)

    plot_max = r_max * 1.18
    ax.set_ylim(0, plot_max)
    ticks = np.arange(0.5, r_max + 1e-6, 0.5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{y:.1f}" for y in ticks], fontsize=5.5, color="#666666")
    ax.set_rlabel_position(20)
    ax.grid(color="#DDDDDD", linewidth=0.5)
    ax.spines["polar"].set_linewidth(0.7)
    ax.spines["polar"].set_color("#333333")

    ax.text(-0.06, 1.08, panel_label, transform=ax.transAxes,
             fontsize=11, fontweight="bold", va="top", ha="left")

    if dft is None:
        ax.text(0.5, 0.5, "No DFT\nreference", transform=ax.transAxes,
                 fontsize=8, color="#B00000", ha="center", va="center")
        return

    theta_fill = np.linspace(0, 2 * np.pi, 200)
    ax.fill_between(theta_fill, 1 - agree_tol, 1 + agree_tol,
                     color="#BBBBBB", alpha=0.25, zorder=0, linewidth=0)

    dft_vals = [1.0] * (n + 1)
    line_dft, = ax.plot(angles, dft_vals, color=COLORS["DFT"], linewidth=1.6,
                         linestyle=LINESTYLES["DFT"], zorder=10, solid_capstyle="round")
    legend_handles.setdefault("DFT", line_dft)

    for pot in POTENTIALS:
        if pot not in mlips:
            continue
        raw = normalized_values(mlips[pot], dft)
        clipped = [max(0.0, min(v, r_max)) for v in raw]
        vals_loop = clipped + clipped[:1]

        line, = ax.plot(
            angles, vals_loop, color=COLORS[pot], linewidth=1.1, linestyle=LINESTYLES[pot],
            marker=MARKERS[pot], markersize=3.2, markerfacecolor=COLORS[pot],
            markeredgecolor="white", markeredgewidth=0.4, zorder=5,
        )
        legend_handles.setdefault(pot, line)

        for ang, raw_v in zip(angles[:-1], raw):
            if raw_v > r_max or raw_v < 0:
                ax.annotate(
                    f"{raw_v:.1f}\u00d7", xy=(ang, r_max), xytext=(ang, r_max * 1.12),
                    fontsize=5, color=COLORS[pot], ha="center", va="center",
                    annotation_clip=False,
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="white")],
                )

    ax.set_title(format_compound_name(compound), fontsize=10, fontweight="bold", pad=16)


def draw_metrics_table(table_ax, table: dict, overall: dict, compounds: list):
    table_ax.axis("off")
    col_labels = [format_compound_name(c) for c in compounds] + ["All"]
    row_labels = [DISPLAY_NAMES[pot] for pot in POTENTIALS]

    cmap = plt.get_cmap("RdYlGn_r")
    vmax = 20.0

    def color_for(v):
        if v is None:
            return "#F2F2F2"
        return cmap(np.clip(v / vmax, 0, 1))

    def text_for(v):
        return "\u2014" if v is None else f"{v:.1f}"

    cell_text, cell_colors = [], []
    for pot in POTENTIALS:
        row_vals = [table[pot].get(c) for c in compounds] + [overall.get(pot)]
        cell_text.append([text_for(v) for v in row_vals])
        cell_colors.append([color_for(v) for v in row_vals])

    tbl = table_ax.table(
        cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
        cellColours=cell_colors, cellLoc="center", rowLoc="right",
        loc="upper center", bbox=[0.16, 0.15, 0.84, 0.85],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)

    n_cols = len(col_labels)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_text_props(fontweight="bold", fontsize=6.5)
            cell.set_facecolor("white")
        elif c == -1:
            cell.set_text_props(fontweight="bold", ha="right", fontsize=6.5,
                                 color=COLORS[POTENTIALS[r - 1]])
            cell.set_facecolor("white")
        elif c == n_cols - 1:
            cell.set_text_props(fontweight="bold", fontsize=6.5)

    table_ax.text(0.0, 1.08, "Mean absolute error vs. DFT (%)", transform=table_ax.transAxes,
                  fontsize=7.5, fontweight="bold", ha="left")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Elastic_constants")
    p.add_argument("--compounds", nargs="+", default=["YB2", "YB4", "YB12"])
    p.add_argument("--outdir", default="Elastic_constants/spider_plots")
    p.add_argument("--rmax", type=float, default=2.0,
                    help="Shared radial scale across all panels (fair visual comparison)")
    p.add_argument("--agree-tol", type=float, default=0.15,
                    help="Half-width of the shaded 'close to DFT' band (0.15 = +/-15%%)")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = p.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_panels = len(args.compounds)
    panel_w = 5.3
    panel_h = panel_w / n_panels * 1.25 if n_panels <= 2 else 3.0
    table_h = 0.24 * (len(POTENTIALS) + 1)
    legend_h = 0.42
    header_h = 0.55
    fig_w = max(7.2, 2.4 * n_panels)
    body_h = panel_h + table_h
    fig_h = body_h + header_h + legend_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    top_frac = (0.88 * body_h + header_h + legend_h) / fig_h
    bottom_frac = legend_h / fig_h

    gs = fig.add_gridspec(2, n_panels, height_ratios=[panel_h, table_h], hspace=0.4, wspace=0.55)
    axes = [fig.add_subplot(gs[0, i], projection="polar") for i in range(n_panels)]
    table_ax = fig.add_subplot(gs[1, :])

    legend_handles = {}
    for i, (ax, compound) in enumerate(zip(axes, args.compounds)):
        plot_panel(ax, base, compound, args.rmax, args.agree_tol, legend_handles, PANEL_LABELS[i])

    mape_table, mape_overall = compute_mape_table(base, args.compounds)
    draw_metrics_table(table_ax, mape_table, mape_overall, args.compounds)

    fig.subplots_adjust(left=0.04, right=0.96, top=top_frac, bottom=bottom_frac)

    fig.canvas.draw()
    subtitle_y = min(0.99, top_frac + 0.06)
    fig.text(0.5, subtitle_y,
              f"Elastic properties normalized to DFT (shaded band: \u00b1{int(args.agree_tol*100)}%)",
              ha="center", fontsize=7.5, color="#444444")

    order = ["DFT"] + POTENTIALS
    handles, labels = [], []
    for k in order:
        if k not in legend_handles:
            continue
        handles.append(legend_handles[k])
        if k == "DFT":
            labels.append(DISPLAY_NAMES[k])
        else:
            mape = mape_overall.get(k)
            suffix = f" ({mape:.1f}% MAPE)" if mape is not None else ""
            labels.append(f"{DISPLAY_NAMES[k]}{suffix}")

    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                ncol=3, frameon=False, fontsize=7.5, handlelength=2.2,
                columnspacing=1.4, handletextpad=0.5)

    stem = outdir / "All_Compounds_Elastic"
    for fmt in args.formats:
        dpi = 400 if fmt == "png" else None
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight",
                     pad_inches=0.08, facecolor="white")
        print(f"wrote {stem.with_suffix(f'.{fmt}')}")
    plt.close(fig)


if __name__ == "__main__":
    main()

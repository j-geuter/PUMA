"""
Plot ELBO curves from eval_elbo.py output files.

Usage:
    python plot_elbo.py
    python plot_elbo.py --files results/elbo_tinygsm-puma.txt results/elbo_tinygsm-baseline.txt
"""
import os, sys, tempfile, argparse
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_PUMA = os.path.dirname(_HERE)

mpl.rcParams.update({
    "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 9, "text.usetex": True, "font.family": "serif",
    "axes.linewidth": 1.0,
})
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]

NAME_MAP = {
    "tinygsm-baseline": r"MDM",
    "tinygsm-puma":     r"\textbf{PUMA}",
}
COLOR_MAP = {
    "tinygsm-baseline": "#FF4D4D",
    "tinygsm-puma":     "#6B8FE8",
}
FALLBACK_COLORS = ["#4169E1", "#FF0000", "#228B22", "#FF8C00", "#9400D3", "#00CED1"]


def read_elbo(path):
    label = "unknown"
    steps, vals = [], []
    in_data = False
    for line in open(path):
        line = line.strip()
        if line.startswith("label:"):
            label = line.split(":", 1)[1].strip()
        if line.startswith("---"):
            in_data = True
            continue
        if in_data and line:
            parts = line.split()
            if len(parts) == 2:
                try:
                    steps.append(int(parts[0]))
                    vals.append(float(parts[1]))
                except ValueError:
                    pass
    return label, np.array(steps), np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+",
                    default=[
                        os.path.join(_PUMA, "results/elbo_tinygsm-puma.txt"),
                        os.path.join(_PUMA, "results/elbo_tinygsm-baseline.txt"),
                    ])
    ap.add_argument("--output", default=os.path.join(_PUMA, "../results/fig_elbo"))
    ap.add_argument("--figsize", nargs=2, type=float, default=[4.25, 2.75])
    args = ap.parse_args()

    curves = []
    for f in args.files:
        label, steps, vals = read_elbo(f)
        curves.append((label, steps, vals))

    fig, ax = plt.subplots(figsize=args.figsize, dpi=300)

    for i, (label, steps, vals) in enumerate(curves):
        color = COLOR_MAP.get(label, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        display = NAME_MAP.get(label, label)
        ax.plot(steps, vals, color=color, lw=2.0, label=display)

    ax.set_ylabel("NELBO")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    max_step = max(s[-1] for _, s, _ in curves)
    ticks = np.arange(200000, max_step + 1, 200000)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{s // 1000}k" for s in ticks])

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("serif")
        lbl.set_fontweight("bold")

    ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.9, alpha=0.65)
    ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], color=COLOR_MAP.get(label, FALLBACK_COLORS[i % len(FALLBACK_COLORS)]),
               lw=3.2, label=NAME_MAP.get(label, label))
        for i, (label, _, _) in enumerate(curves)
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.9,
              edgecolor="0.8", handlelength=2.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(f"{args.output}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{args.output}.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved {args.output}.png and {args.output}.pdf")


if __name__ == "__main__":
    main()

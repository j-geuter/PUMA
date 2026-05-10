#!/usr/bin/env python3
"""
Plot trajectory distance metrics from traj_log.npy files produced during training.

Usage:
    python plot_traj_distance.py --traj_path /path/to/traj_log.npy --out_dir results
"""
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def metric_abs(u_t, u_T, pm):
    """Mean absolute difference in unmask order, excluding prompt positions."""
    B = u_t.shape[0]
    diffs = []
    for b in range(B):
        mask = ~pm[b]
        n = mask.sum()
        if n == 0:
            continue
        diffs.append(np.mean(np.abs(u_t[b, mask].astype(np.float32)
                                    - u_T[b, mask].astype(np.float32))))
    return float(np.mean(diffs)) if diffs else 0.0


def metric_abs_norm(u_t, u_T, pm):
    """Absolute difference after per-sample [0,1] normalization."""
    B = u_t.shape[0]
    diffs = []
    for b in range(B):
        mask = ~pm[b]
        if mask.sum() == 0:
            continue
        ut_b = u_t[b, mask].astype(np.float32)
        uT_b = u_T[b, mask].astype(np.float32)
        ut_max = max(ut_b.max(), 1.0)
        uT_max = max(uT_b.max(), 1.0)
        diffs.append(np.mean(np.abs(ut_b / ut_max - uT_b / uT_max)))
    return float(np.mean(diffs)) if diffs else 0.0


def metric_kendall_custom(u_t, u_T, pm):
    """Fraction of position pairs with disagreeing unmask order."""
    B = u_t.shape[0]
    rates = []
    for b in range(B):
        mask = ~pm[b]
        if mask.sum() < 2:
            continue
        ut_b = u_t[b, mask]
        uT_b = u_T[b, mask]
        N = len(ut_b)
        p, q = np.triu_indices(N, k=1)
        dis1 = (ut_b[p] <= ut_b[q]) & (uT_b[p] > uT_b[q])
        dis2 = (ut_b[q] <= ut_b[p]) & (uT_b[q] > uT_b[p])
        rates.append(float((dis1 | dis2).mean()))
    return float(np.mean(rates)) if rates else 0.0


def rcparams_style(usetex=False):
    mpl.rcParams.update({
        "text.usetex": bool(usetex),
        "font.family": "serif",
        "axes.linewidth": 1.0,
    })
    if usetex:
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{newtxtext,newtxmath}"


def plot_series(x, y, xlabel, ylabel, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.3), dpi=300)
    ax.plot(x, y, marker=None, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_path", type=str, required=True,
                    help="Path to traj_log.npy")
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--prefix", type=str, default="traj_dist")
    ap.add_argument("--usetex", action="store_true")
    args = ap.parse_args()

    rcparams_style(args.usetex)
    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(args.traj_path, allow_pickle=True).item()
    pm = data["prompt_masks"].astype(bool)
    orders = data["unmask_orders"]

    steps = sorted(orders.keys())
    print(f"Found {len(steps)} logged steps: {steps[0]} ... {steps[-1]}")

    if len(steps) < 2:
        print("Need at least 2 logged steps to compute distances.")
        return

    final_step = steps[-1]
    u_T = orders[final_step]

    y_abs, y_abs_norm, y_kendall = [], [], []
    for s in steps:
        u_t = orders[s]
        y_abs.append(metric_abs(u_t, u_T, pm))
        y_abs_norm.append(metric_abs_norm(u_t, u_T, pm))
        y_kendall.append(metric_kendall_custom(u_t, u_T, pm))

    x = np.array(steps, dtype=np.int64)

    plot_series(x, y_abs, "Training step", "Trajectory distance",
                os.path.join(args.out_dir, f"{args.prefix}_abs.pdf"))
    plot_series(x, y_abs, "Training step", "Trajectory distance",
                os.path.join(args.out_dir, f"{args.prefix}_abs.svg"))

    plot_series(x, y_abs_norm, "Training step", "Trajectory distance (normalized)",
                os.path.join(args.out_dir, f"{args.prefix}_abs_norm01.pdf"))
    plot_series(x, y_abs_norm, "Training step", "Trajectory distance (normalized)",
                os.path.join(args.out_dir, f"{args.prefix}_abs_norm01.svg"))

    plot_series(x, y_kendall, "Training step", "Kendall trajectory distance",
                os.path.join(args.out_dir, f"{args.prefix}_kendall_custom.pdf"))
    plot_series(x, y_kendall, "Training step", "Kendall trajectory distance",
                os.path.join(args.out_dir, f"{args.prefix}_kendall_custom.svg"))

    print(f"\nFinal step: {final_step} (last point should be ~0 for all metrics)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plot_traj_from_log.py

Load a traj_log.npy produced by plot_traj_from_ckpts.py (or train.py) and:
  - compute trajectory distance metrics vs. the final checkpoint
  - save plots (PDF + PNG) for each metric
  - save a .txt table of the values

Usage:
    python plot_traj_from_log.py --traj_log /path/to/traj_log.npy
"""
import os, sys, tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import argparse
from typing import List
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm


# ── metrics ───────────────────────────────────────────────────────────────────

def extract_nonprompt(orders: np.ndarray, prompt_mask_np: np.ndarray) -> List[np.ndarray]:
    return [orders[b, ~prompt_mask_np[b]] for b in range(orders.shape[0])]


def metric_abs(u_t_list, u_T_list) -> float:
    diffs = [np.abs(a.astype(np.float32) - b.astype(np.float32))
             for a, b in zip(u_t_list, u_T_list)]
    return float(np.mean(np.concatenate(diffs)))


def metric_abs_norm(u_t_list, u_T_list) -> float:
    vals = []
    for u_t, u_T in zip(u_t_list, u_T_list):
        ut_n = u_t.astype(np.float32) / max(u_t.max(), 1)
        uT_n = u_T.astype(np.float32) / max(u_T.max(), 1)
        vals.append(np.abs(ut_n - uT_n))
    return float(np.mean(np.concatenate(vals)))


def metric_kendall_custom(u_t_list, u_T_list) -> float:
    dis_sum = 0.0
    pair_count = 0
    for u_t, u_T in zip(u_t_list, u_T_list):
        N = len(u_t)
        if N < 2:
            continue
        p, q = np.triu_indices(N, k=1)
        dis = ((u_t[p] <= u_t[q]) & (u_T[p] > u_T[q])) | \
              ((u_t[q] <= u_t[p]) & (u_T[q] > u_T[p]))
        dis_sum += dis.sum()
        pair_count += len(p)
    return float(dis_sum / pair_count) if pair_count > 0 else 0.0


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_series(x, y, xlabel, ylabel, out_stem):
    mpl.rcParams.update({"font.family": "serif", "axes.linewidth": 1.0})
    fig, ax = plt.subplots(figsize=(4.8, 3.3), dpi=300)
    ax.plot(x, y, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    fig.savefig(out_stem + ".png", bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_log", required=True, help="Path to traj_log.npy")
    ap.add_argument("--out_dir",
                    default="/n/holylabs/LABS/dam_lab/Users/jgeuter/MDMPre/jay_mdm_playground/PUMA/results",
                    help="Output directory for plots and .txt")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.traj_log} ...")
    traj_log = np.load(args.traj_log, allow_pickle=True).item()

    orders = traj_log["unmask_orders"]
    pm_np = traj_log["prompt_mask"]

    sorted_steps = sorted(orders.keys())
    print(f"Found {len(sorted_steps)} logged steps: {sorted_steps[0]} ... {sorted_steps[-1]}")

    u_T_nonprompt = extract_nonprompt(orders[sorted_steps[-1]], pm_np)

    y_abs, y_abs_norm, y_kendall = [], [], []
    for s in tqdm(sorted_steps, desc="Metrics"):
        u_t_np = extract_nonprompt(orders[s], pm_np)
        y_abs.append(metric_abs(u_t_np, u_T_nonprompt))
        y_abs_norm.append(metric_abs_norm(u_t_np, u_T_nonprompt))
        y_kendall.append(metric_kendall_custom(u_t_np, u_T_nonprompt))

    x = np.array(sorted_steps, dtype=np.int64)

    # save plots
    plot_series(x, y_abs,      "Training step", "Trajectory distance",
                os.path.join(args.out_dir, "traj_dist_abs"))
    plot_series(x, y_abs_norm, "Training step", "Trajectory distance",
                os.path.join(args.out_dir, "traj_dist_abs_norm01"))
    plot_series(x, y_kendall,  "Training step", "Kendall trajectory distance",
                os.path.join(args.out_dir, "traj_dist_kendall_custom"))

    # save .txt table
    txt_path = os.path.join(args.out_dir, "traj_dist_values.txt")
    with open(txt_path, "w") as f:
        f.write(f"# traj_log: {args.traj_log}\n")
        f.write(f"# step\tabs\tabs_norm01\tkendall_custom\n")
        for step, a, an, k in zip(sorted_steps, y_abs, y_abs_norm, y_kendall):
            f.write(f"{step}\t{a:.6f}\t{an:.6f}\t{k:.6f}\n")

    print(f"Saved to {args.out_dir}/")
    print(f"  traj_dist_abs.{{pdf,png}}")
    print(f"  traj_dist_abs_norm01.{{pdf,png}}")
    print(f"  traj_dist_kendall_custom.{{pdf,png}}")
    print(f"  traj_dist_values.txt")


if __name__ == "__main__":
    main()

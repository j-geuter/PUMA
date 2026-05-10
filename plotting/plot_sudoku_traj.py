#!/usr/bin/env python3
"""
plot_sudoku_traj.py

Compute and plot sudoku training-time unmasking trajectories across training
checkpoints.  For each EMA checkpoint, runs create_training_traj (K-stage
progressive unmasking with confidence collapse) to record the unmask order,
then visualises how that order evolves over training for a single selected puzzle.
"""
import os, sys, tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import argparse
import glob
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import PowerNorm
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_PUMA = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PUMA)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PUMA)
sys.path.insert(0, _ROOT)

from plot_traj_from_ckpts import (
    load_model, load_samples, create_training_traj,
    unmask_order_from_track as unmask_order_from_track_torch,
    get_final_k,
)
from plot_sudoku import rcparams_style, FONT_SIZE_DEFAULT

# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CKPT_DIR = (
    "/n/netscratch/dam_lab/Lab/jgeuter/MDMPre/ckpts/date=2026-04-08-15-37"
)
DEFAULT_OUT_DIR = os.path.join(_PUMA, "results")
DEFAULT_NUM_SAMPLES = 100
DEFAULT_STEPS = [0, 10_000, 20_000, 30_000]  # + final (appended automatically)

GIVEN_FACECOLOR = "#5B9E5B"  # lightly darkish green for prompt/given cells

# ── modified draw_sudoku (adds given_facecolor) ─────────────────────────────

def draw_sudoku(
    ax,
    digits_flat: np.ndarray,        # (81,)
    gt_flat: np.ndarray,            # (81,)
    given_mask_flat: np.ndarray,    # (81,) True for given/prompt cells
    order_flat: np.ndarray,         # (81,) unmask order
    mask_id: int,
    cmap,
    norm,
    title: str,
    given_facecolor=GIVEN_FACECOLOR,
    masked_facecolor="lightgray",
    digit_fontsize=FONT_SIZE_DEFAULT,
):
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12)

    for r in range(9):
        for c in range(9):
            idx = r * 9 + c
            is_given = bool(given_mask_flat[idx])

            d_pred = int(digits_flat[idx])
            d_gt = int(gt_flat[idx])
            d_disp = d_gt if is_given else d_pred

            if is_given:
                face = given_facecolor
            elif d_disp == mask_id:
                face = masked_facecolor
            else:
                face = cmap(norm(float(order_flat[idx])))

            rect = Rectangle(
                (c, 8 - r), 1, 1,
                linewidth=1, edgecolor="black", facecolor=face,
            )
            ax.add_patch(rect)

            if d_disp != mask_id:
                ax.text(
                    c + 0.5, 8 - r + 0.5, str(d_disp),
                    ha="center", va="center",
                    fontsize=digit_fontsize,
                    color="white",
                    fontfamily="Nimbus Roman",
                    fontweight="normal",
                )

    for k in range(4):
        ax.axvline(x=3 * k, color="black", linewidth=3)
        ax.axhline(y=3 * k, color="black", linewidth=3)


# ── training trajectory computation ──────────────────────────────────────────

def run_training_traj_single_ckpt(
    model, x0, prompt_mask, mask_id, K, mode, confidence_threshold, device, micro_bs,
):
    """Run create_training_traj on all samples with OOM-safe micro-batching.
    Returns unmask_order (B, L) int32.
    """
    B = x0.shape[0]
    all_orders = []

    for start in range(0, B, micro_bs):
        end = min(start + micro_bs, B)
        track_list = create_training_traj(
            model,
            x0[start:end],
            prompt_mask[start:end],
            mask_id=mask_id,
            K=K,
            mode=mode,
            confidence_threshold=confidence_threshold,
            device=device,
            track=True,
        )
        order = unmask_order_from_track_torch(track_list, mask_id)
        all_orders.append(order)

    return np.concatenate(all_orders, axis=0)  # (B, L)


def compute_all_trajectories(ckpt_dir, num_samples, device, save_path):
    # discover EMA checkpoints
    ckpt_files = sorted(
        glob.glob(os.path.join(ckpt_dir, "ema_step=*.pt")),
        key=lambda p: int(os.path.basename(p).split("=")[1].split(".")[0]),
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No ema_step=*.pt files in {ckpt_dir}")

    steps = [int(os.path.basename(p).split("=")[1].split(".")[0]) for p in ckpt_files]
    print(f"Found {len(ckpt_files)} EMA checkpoints: steps {steps}")

    # load config from first checkpoint
    first_ckpt = torch.load(ckpt_files[0], map_location="cpu")
    cfg = first_ckpt["config"]
    cfg_data = cfg["data"]
    cfg_train = cfg["training"]
    mask_id = int(cfg_data["mask_id"])
    K = get_final_k(cfg_train)
    mode = cfg_train.get("mode", "standard")
    confidence_threshold = float(cfg_train.get("confidence_threshold", 0.9))
    del first_ckpt

    print(f"Training config: K={K}, mode={mode}, confidence_threshold={confidence_threshold}")

    # load fixed test samples
    print(f"Loading {num_samples} fixed test samples...")
    x0, prompt_mask = load_samples(cfg_data, num_samples, device)
    x0_np = x0.cpu().numpy()
    pm_np = prompt_mask.cpu().numpy()

    # resume from partial save
    if os.path.exists(save_path):
        traj_data = np.load(save_path, allow_pickle=True).item()
        unmask_orders = traj_data["unmask_orders"]
        print(f"Resuming: {len(unmask_orders)}/{len(ckpt_files)} steps already computed")
    else:
        unmask_orders = {}
        traj_data = {
            "x0": x0_np,
            "prompt_mask": pm_np,
            "mask_id": mask_id,
            "K": K,
            "mode": mode,
            "unmask_orders": unmask_orders,
        }

    micro_bs = num_samples

    for ckpt_path, step in tqdm(list(zip(ckpt_files, steps)), desc="Checkpoints"):
        if step in unmask_orders:
            continue

        model = load_model(ckpt_path, device)

        # OOM retry
        while True:
            try:
                orders = run_training_traj_single_ckpt(
                    model, x0, prompt_mask, mask_id,
                    K, mode, confidence_threshold, device, micro_bs,
                )
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if micro_bs <= 1:
                    raise RuntimeError("OOM even with micro_batch_size=1")
                micro_bs = max(1, micro_bs // 2)
                tqdm.write(f"  OOM at step {step}, halving to micro_bs={micro_bs}")

        unmask_orders[step] = orders
        np.save(save_path, traj_data)
        tqdm.write(f"  step {step} done, saved to {save_path}")

        del model
        torch.cuda.empty_cache()

    return traj_data


# ── best sample selection ────────────────────────────────────────────────────

def select_best_sample(traj_data):
    """Select sample whose unmask ordering converges earliest to the final
    checkpoint's ordering.  Pick the one with minimum sum-of-squared-distances
    to the final ordering across all earlier checkpoints (non-prompt cells only).
    Training trajectories always use GT tokens, so no correctness filter needed.
    """
    unmask_orders = traj_data["unmask_orders"]
    pm = traj_data["prompt_mask"].astype(bool)
    B = traj_data["x0"].shape[0]

    sorted_steps = sorted(unmask_orders.keys())
    final_step = sorted_steps[-1]

    order_final = unmask_orders[final_step]
    non_prompt = ~pm

    cost = np.zeros(B, dtype=np.float64)
    for step in sorted_steps[:-1]:
        diff = unmask_orders[step].astype(np.float64) - order_final.astype(np.float64)
        cost += (diff ** 2 * non_prompt).sum(axis=1)

    best_idx = int(np.argmin(cost))
    return best_idx, final_step


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_sudoku_traj(traj_data, best_idx, display_steps, out_prefix, usetex=False):
    rcparams_style(usetex)

    unmask_orders = traj_data["unmask_orders"]
    x0 = traj_data["x0"]
    pm = traj_data["prompt_mask"].astype(bool)
    mask_id = traj_data["mask_id"]

    givens_mask = pm[best_idx]         # (81,)
    gt = x0[best_idx].astype(np.int64) # (81,)

    # colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "unmask_rb", ["royalblue", "orangered"], N=256,
    )
    # vmax: max unmask-order value across displayed panels (non-prompt cells)
    all_vmax = 0
    for step in display_steps:
        order = unmask_orders[step][best_idx]
        non_prompt_vals = order[~givens_mask]
        if non_prompt_vals.size > 0:
            all_vmax = max(all_vmax, int(non_prompt_vals.max()))
    vmax = max(all_vmax, 1)
    norm = PowerNorm(gamma=0.45, vmin=0, vmax=vmax, clip=True)

    sorted_steps = sorted(unmask_orders.keys())
    final_step = sorted_steps[-1]

    ncols = len(display_steps)
    fig_w = 2.6 * ncols + 0.8
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 3.2), dpi=300)
    if ncols == 1:
        axes = [axes]

    for k, step in enumerate(display_steps):
        order_step = unmask_orders[step][best_idx]

        title = f"Step {step:,}"
        if step == final_step:
            title += " (final)"

        # training trajectories always reveal GT tokens
        draw_sudoku(
            axes[k],
            digits_flat=gt,
            gt_flat=gt,
            given_mask_flat=givens_mask,
            order_flat=order_step,
            mask_id=mask_id,
            cmap=cmap,
            norm=norm,
            title=title,
        )

    plt.subplots_adjust(left=0.02, right=0.90, top=0.90, bottom=0.05, wspace=0.20)
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    cb = ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_ticks([0, vmax])
    cb.set_ticklabels(["early", "late"])
    cb.ax.tick_params(labelsize=11)
    cb.set_label("Unmask time", fontsize=12)

    fig.savefig(f"{out_prefix}.png", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_prefix}.{{png,pdf}}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_steps_arg(s, available_steps):
    """Parse --steps or return default [0, 10k, 20k, 30k, final]."""
    final = max(available_steps)
    if s is None:
        steps = [st for st in DEFAULT_STEPS if st in available_steps]
        if final not in steps:
            steps.append(final)
        return steps
    parsed = [int(x) for x in s.replace(",", " ").split()]
    return parsed


def main():
    ap = argparse.ArgumentParser(
        description="Compute and plot sudoku unmasking trajectories across training checkpoints.",
    )
    ap.add_argument("--input", type=str, default=None,
                    help="Precomputed .npy file (skip computation)")
    ap.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    ap.add_argument("--steps", type=str, default=None,
                    help="Checkpoint steps to display (default: 0,10k,20k,30k,final)")
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--out_name", type=str, default="sudoku_traj")
    ap.add_argument("--save_name", type=str, default="sudoku_traj_data.npy")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--usetex", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, args.save_name)

    # Phase 1: compute or load
    if args.input is not None:
        print(f"Loading precomputed data from {args.input}")
        traj_data = np.load(args.input, allow_pickle=True).item()
    else:
        traj_data = compute_all_trajectories(
            args.ckpt_dir, args.num_samples, device, save_path,
        )

    # Phase 2: select best sample
    best_idx, final_step = select_best_sample(traj_data)
    traj_data["best_idx"] = best_idx
    if args.input is None:
        np.save(save_path, traj_data)
    print(f"Selected sample idx={best_idx}")

    # Phase 3: plot
    available_steps = sorted(traj_data["unmask_orders"].keys())
    display_steps = parse_steps_arg(args.steps, available_steps)
    for s in display_steps:
        if s not in traj_data["unmask_orders"]:
            raise ValueError(
                f"Step {s} not in computed data. Available: {available_steps}"
            )

    out_prefix = os.path.join(args.out_dir, args.out_name)
    plot_sudoku_traj(traj_data, best_idx, display_steps, out_prefix, args.usetex)
    print(f"Best sample idx={best_idx}, displayed steps={display_steps}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plot_traj_from_ckpts.py

Compute trajectory distances from a checkpoint directory and plot them.
Uses the training-faithful K-stage progressive unmasking with confidence collapse,
matching what the model experienced during training.

Works for both tinygsm and sudoku (single/double sequence).

Usage:
    python plot_traj_from_ckpts.py --ckpt_dir /path/to/ckpt/dir
"""
import math
import os, sys, tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import argparse
import glob
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_PUMA = os.path.dirname(_HERE)
sys.path.insert(0, _PUMA)
from model.transformer import MDMTransformer, MDMConfig


# ── training-faithful trajectory computation ──────────────────────────────────
# Adapted from create_training_traj in sudoku branch train.py,
# made dataset-agnostic (accepts full x0 directly, no hardcoded [:, 81:]).

@torch.no_grad()
def create_training_traj(
    model,
    x0: torch.Tensor,           # (B, L) full GT tokens
    prompt_mask: torch.Tensor,  # (B, L) bool, True = prompt (never masked)
    mask_id: int,
    K: int,
    mode: str = "standard",
    confidence_threshold: float = 0.9,
    device: Optional[torch.device] = None,
    track: bool = True,
) -> List[torch.Tensor]:
    """
    Simulate a PUMA training trajectory using GT tokens.
    Returns track_list: list of (B, L) CPU tensors, one snapshot per stage.
    """
    if device is None:
        device = x0.device

    B, L = x0.shape
    x0_full = x0.to(device)
    prompt_mask = prompt_mask.to(device)

    # initialize xt: prompt positions kept, answer positions masked
    xt = x0_full.clone()
    xt[~prompt_mask] = mask_id

    # K equal intervals [i/K, (i+1)/K)
    edges = torch.linspace(0.0, 1.0, K + 1, device=device)
    lower, upper = edges[:-1], edges[1:]

    L_eff = (~prompt_mask).sum(dim=1).long()  # (B,) non-prompt count per sample

    def sample_ratio(stages: torch.Tensor) -> torch.Tensor:
        lo = lower.index_select(0, stages)
        hi = upper.index_select(0, stages)
        return lo + torch.rand_like(lo) * (hi - lo)

    def sample_target_unmasked(ratio: torch.Tensor) -> torch.Tensor:
        num = torch.round(ratio * L_eff.float()).long()
        num = torch.minimum(num, (L_eff - 1).clamp_min(1))
        return num

    def unmask_topk(scores: torch.Tensor, to_reveal: torch.Tensor):
        for b in range(B):
            k = int(to_reveal[b].item())
            if k <= 0:
                continue
            k = min(k, int(scores.shape[1]))
            _, idx = torch.topk(scores[b], k=k)
            xt[b, idx] = x0_full[b, idx]

    def calculate_phase() -> torch.Tensor:
        current = (~prompt_mask & (xt != mask_id)).sum(dim=1).long()
        ratio_now = current.float() / L_eff.clamp_min(1).float()
        stage = torch.bucketize(ratio_now, upper[:-1])
        return stage.clamp_(0, K - 1).long()

    # Stage 0: random initialization — reveal u0 GT tokens randomly
    phase = torch.zeros(B, dtype=torch.long, device=device)
    ratio0 = sample_ratio(phase)
    u0 = sample_target_unmasked(ratio0)

    rand_score = torch.rand((B, L), device=device)
    rand_score = torch.where((xt == mask_id) & (~prompt_mask), rand_score,
                             torch.full_like(rand_score, float('-inf')))
    avail = ((xt == mask_id) & (~prompt_mask)).sum(dim=1).long()
    u0 = torch.minimum(u0, avail)
    unmask_topk(rand_score, u0)

    track_list: List[torch.Tensor] = []
    if track:
        track_list.append(xt.clone().detach().cpu())

    # K-1 progressive stages
    for _ in range(K - 1):
        if ((xt == mask_id) & (~prompt_mask)).sum().item() == 0:
            break

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            logits = model(xt)
        log_probs = F.log_softmax(logits, dim=-1)

        phase_next = (phase + 1) % K
        replace = (phase_next == 0)

        mask_idx = (xt == mask_id)
        score_conf = log_probs.max(dim=-1).values
        score_conf = torch.where(mask_idx & (~prompt_mask), score_conf,
                                 torch.full_like(score_conf, float('-inf')))

        ratio = sample_ratio(phase_next)
        num_unmask = sample_target_unmasked(ratio)
        current_unmasked = ((~mask_idx) & (~prompt_mask)).sum(dim=1).long()
        to_reveal = (num_unmask - current_unmasked).clamp_min(0)
        to_reveal = torch.where(replace, torch.zeros_like(to_reveal), to_reveal)

        if int(to_reveal.max().item()) > 0:
            avail = ((xt == mask_id) & (~prompt_mask)).sum(dim=1).long()
            to_reveal = torch.minimum(to_reveal, avail)
            unmask_topk(score_conf, to_reveal)

        if mode == "confidence_collapse":
            tau = math.log(float(confidence_threshold))
            pmax = log_probs.max(dim=-1).values
            collapse = (pmax > tau) & (xt == mask_id) & (~prompt_mask)
            xt = torch.where(collapse, x0_full, xt)
            phase_next = calculate_phase()

        phase = phase_next

        if track:
            track_list.append(xt.clone().detach().cpu())

        if bool(replace.all().item()):
            break

    return track_list


def unmask_order_from_track(track_list: List[torch.Tensor], mask_id: int) -> np.ndarray:
    """
    track_list: list of (B, L) tensors (CPU).
    Returns (B, L) int32 array where each value is the first track step
    at which that position was unmasked (T if never unmasked).
    """
    tr = np.stack([t.numpy() for t in track_list], axis=0)  # (T, B, L)
    unmasked = (tr != mask_id)
    any_unmasked = unmasked.any(axis=0)
    first = unmasked.argmax(axis=0).astype(np.int32)
    T = tr.shape[0]
    return np.where(any_unmasked, first, T).astype(np.int32)


def compute_traj_order_microbatch(
    model, x0: torch.Tensor, prompt_mask: torch.Tensor,
    mask_id: int, K: int, mode: str, confidence_threshold: float,
    device, micro_batch_size: int
) -> np.ndarray:
    """Run create_training_traj in micro-batches, return (B, L) unmask order."""
    B = x0.shape[0]
    results = []
    for start in range(0, B, micro_batch_size):
        end = min(start + micro_batch_size, B)
        track = create_training_traj(
            model, x0[start:end], prompt_mask[start:end],
            mask_id=mask_id, K=K, mode=mode,
            confidence_threshold=confidence_threshold, device=device, track=True,
        )
        results.append(unmask_order_from_track(track, mask_id))
    return np.concatenate(results, axis=0)


# ── metrics (over non-prompt positions only) ──────────────────────────────────

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


# ── config helpers ────────────────────────────────────────────────────────────

def get_final_k(train_cfg: dict) -> int:
    ks = train_cfg.get("k_schedule")
    return int(ks[-1][0]) if ks else int(train_cfg["K"])


# ── model / data loading ──────────────────────────────────────────────────────

def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model_cfg_dict = dict(cfg["model"])
    arm_init_path = model_cfg_dict.get("arm_init", "none")
    model_config = MDMConfig(**model_cfg_dict)
    if arm_init_path != "none":
        model_config.predict_next_token = True
    model = MDMTransformer(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def load_samples(cfg_data: dict, num_samples: int, device):
    dataset = cfg_data["dataset"]
    data_dir = cfg_data["data_dir"]

    if dataset == "sudoku":
        single_seq = cfg_data.get("single_seq_sudoku", False)
        raw = np.load(os.path.join(data_dir, "test_mdm.npy"))[:num_samples]
        if not single_seq:
            x0 = torch.from_numpy(raw).long().to(device)
            pm = torch.zeros(x0.shape, dtype=torch.bool, device=device)
            pm[:, :81] = True
        else:
            puzzle = raw[:, :81]
            x0 = torch.from_numpy(raw[:, 81:]).long().to(device)
            pm = torch.from_numpy(puzzle != 0).bool().to(device)
    else:  # tinygsm
        from data.tiny_gsm import split_tinygsm
        val_ratio = cfg_data.get("val_ratio", 0.02)
        seed = cfg_data.get("seed", 2026)
        _, val_ds = split_tinygsm(data_dir, val_ratio=val_ratio, seed=seed)
        n = min(num_samples, len(val_ds))
        samples = [val_ds[i] for i in range(n)]
        x0 = torch.stack([s["labels"] for s in samples]).to(device)
        pm = torch.stack([s["prompt_mask"] for s in samples]).bool().to(device)

    return x0, pm


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
    ap.add_argument("--ckpt_dir", required=True,
                    help="Directory containing step=*.pt checkpoints")
    ap.add_argument("--out_dir",
                    default="/n/holylabs/LABS/dam_lab/Users/jgeuter/MDMPre/jay_mdm_playground/PUMA/results",
                    help="Output directory for plots")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--stride", type=int, default=4,
                    help="Use every p-th checkpoint (always includes first and last)")
    ap.add_argument("--traj_log", default=None,
                    help="Trajectory log path (default: {ckpt_dir}/traj_log.npy)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir
    traj_log_path = args.traj_log or os.path.join(args.ckpt_dir, "traj_log.npy")
    os.makedirs(out_dir, exist_ok=True)

    # gather and sort all checkpoints by step number
    all_ckpt_files = sorted(
        glob.glob(os.path.join(args.ckpt_dir, "step=*.pt")),
        key=lambda p: int(os.path.basename(p).split("=")[1].split(".")[0])
    )
    if not all_ckpt_files:
        raise FileNotFoundError(f"No step=*.pt files found in {args.ckpt_dir}")

    # Build virtual list: [step=0 (random init)] + all_ckpt_files
    # Apply stride over the combined virtual list (length N+1), always include
    # virtual index 0 (step=0) and virtual index N (last checkpoint).
    N = len(all_ckpt_files)
    virtual_indices = list(range(0, N + 1, args.stride))
    if N not in virtual_indices:
        virtual_indices.append(N)
    virtual_indices = sorted(set(virtual_indices))

    include_step0 = (0 in virtual_indices)
    ckpt_files = [all_ckpt_files[i - 1] for i in virtual_indices if i > 0]
    steps = [int(os.path.basename(p).split("=")[1].split(".")[0]) for p in ckpt_files]

    total_selected = (1 if include_step0 else 0) + len(ckpt_files)
    print(f"Checkpoints: {N} total, {total_selected} selected "
          f"(stride={args.stride}, step=0 included, last={steps[-1] if steps else 'n/a'})")

    # read config from first checkpoint
    first_ckpt = torch.load(ckpt_files[0], map_location="cpu")
    cfg = first_ckpt["config"]
    cfg_data = cfg["data"]
    cfg_train = cfg["training"]

    mask_id = int(cfg_data["mask_id"])
    K = get_final_k(cfg_train)
    mode = cfg_train.get("mode", "standard")
    confidence_threshold = float(cfg_train.get("confidence_threshold", 0.9))

    print(f"Dataset: {cfg_data['dataset']}  |  device: {device}  |  "
          f"K={K}  |  mode={mode}  |  confidence_threshold={confidence_threshold}")

    # load fixed samples
    print(f"Loading {args.num_samples} fixed val samples...")
    x0, pm = load_samples(cfg_data, args.num_samples, device)
    x0_np = x0.cpu().numpy()
    pm_np = pm.cpu().numpy()
    avg_nonprompt = (~pm).sum(dim=1).float().mean().item()
    print(f"  x0: {list(x0.shape)}, avg non-prompt tokens/sample: {avg_nonprompt:.1f}")

    # load or init trajectory log
    if os.path.exists(traj_log_path):
        traj_log = np.load(traj_log_path, allow_pickle=True).item()
        orders = traj_log["unmask_orders"]
        print(f"Resuming: {len(orders)}/{len(ckpt_files)} steps already in log")
    else:
        orders = {}
        traj_log = {
            "x0":            x0_np,
            "prompt_mask":   pm_np,
            "mask_id":       mask_id,
            "dataset":       cfg_data["dataset"],
            "K":             K,
            "mode":          mode,
            "unmask_orders": orders,
        }

    # compute unmask orders — adaptive micro-batch OOM handling
    micro_bs = args.num_samples

    def run_traj(model, step_label):
        nonlocal micro_bs
        while True:
            try:
                return compute_traj_order_microbatch(
                    model, x0, pm, mask_id, K, mode, confidence_threshold,
                    device, micro_batch_size=micro_bs,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if micro_bs <= 1:
                    raise RuntimeError("OOM with micro_batch_size=1, cannot proceed.")
                micro_bs = max(1, micro_bs // 2)
                tqdm.write(f"  OOM at step {step_label} — halving micro_batch_size to {micro_bs}")

    # step 0: random-init model
    if include_step0 and 0 not in orders:
        tqdm.write("Computing step=0 (random init)...")
        model_cfg_dict = dict(cfg["model"])
        arm_init_path = model_cfg_dict.get("arm_init", "none")
        model_config = MDMConfig(**model_cfg_dict)
        if arm_init_path != "none":
            model_config.predict_next_token = True
        model = MDMTransformer(model_config).to(device)
        model.eval()
        orders[0] = run_traj(model, 0)
        np.save(traj_log_path, traj_log)
        del model
        torch.cuda.empty_cache()

    for ckpt_path, step in tqdm(list(zip(ckpt_files, steps)), desc="Checkpoints"):
        if step in orders:
            continue

        model = load_model(ckpt_path, device)
        orders[step] = run_traj(model, step)
        np.save(traj_log_path, traj_log)

        del model
        torch.cuda.empty_cache()

    print(f"All {len(orders)} steps computed. Trajectory log: {traj_log_path}")
    print("Calculating metrics...")

    sorted_steps = sorted(orders.keys())
    u_T_nonprompt = extract_nonprompt(orders[sorted_steps[-1]], pm_np)

    y_abs, y_abs_norm, y_kendall = [], [], []
    for s in tqdm(sorted_steps, desc="Metrics"):
        u_t_np = extract_nonprompt(orders[s], pm_np)
        y_abs.append(metric_abs(u_t_np, u_T_nonprompt))
        y_abs_norm.append(metric_abs_norm(u_t_np, u_T_nonprompt))
        y_kendall.append(metric_kendall_custom(u_t_np, u_T_nonprompt))

    x = np.array(sorted_steps, dtype=np.int64)

    plot_series(x, y_abs,      "Training step", "Trajectory distance",
                os.path.join(out_dir, "traj_dist_abs"))
    plot_series(x, y_abs_norm, "Training step", "Trajectory distance",
                os.path.join(out_dir, "traj_dist_abs_norm01"))
    plot_series(x, y_kendall,  "Training step", "Kendall trajectory distance",
                os.path.join(out_dir, "traj_dist_kendall_custom"))

    print(f"Plots saved to {out_dir}/")
    for name in ("traj_dist_abs", "traj_dist_abs_norm01", "traj_dist_kendall_custom"):
        print(f"  {name}.{{pdf,png}}")


if __name__ == "__main__":
    main()

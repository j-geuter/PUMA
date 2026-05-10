#!/usr/bin/env python3
"""
eval_elbo.py — Compute the MDM ELBO across training checkpoints.

For each checkpoint, evaluates the standard absorbing-diffusion ELBO on the
TinyGSM val split (full ground-truth sequences):

    ELBO_estimate = (1/t) * Σ_{i: x_t^i = mask} -log p_θ(x_0^i | x_t)

where t ~ U(ε, 1) and each non-prompt token is masked i.i.d. with prob t.
Reports per-token ELBO averaged over the dataset.

Usage:
    python eval_elbo.py \
        --ckpt_dir /path/to/ckpts/date=... \
        --max_step 420000 --skip 40000
"""
import os, sys, re, gc, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from model.transformer import MDMTransformer, MDMConfig
from data.tiny_gsm import split_tinygsm
from torch.utils.data import DataLoader, Subset


# ── helpers ──────────────────────────────────────────────────────────────────

def scan_checkpoints(ckpt_dir, max_step, skip):
    """Return sorted list of (step, path) for ema checkpoints, filtered."""
    pattern = re.compile(r"ema_step=(\d+)\.pt$")
    entries = []
    for fname in os.listdir(ckpt_dir):
        m = pattern.match(fname)
        if m:
            step = int(m.group(1))
            if step <= max_step:
                entries.append((step, os.path.join(ckpt_dir, fname)))
    entries.sort()
    if not entries:
        return []

    first, last = entries[0][0], entries[-1][0]
    filtered = []
    for step, path in entries:
        if step == first or step == last or step % skip == 0:
            filtered.append((step, path))
    return filtered


def _try_forward(model, masked_input):
    """Isolated frame for OOM safety — returns logits or None."""
    try:
        with torch.no_grad():
            return model(masked_input)
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        gc.collect()


# ── ELBO computation ─────────────────────────────────────────────────────────

def compute_elbo(model, val_data, mask_id, device, batch_size, arm_init, seed,
                 eps=0.001):
    """Compute per-token ELBO over the dataset with adaptive batch size.

    Returns (elbo_per_token, n_samples, final_batch_size).
    """
    total_elbo = 0.0
    total_answer_tokens = 0
    n_samples = 0
    idx = 0  # current position in dataset
    N = len(val_data)

    torch.manual_seed(seed)

    pbar = tqdm(total=N, desc="ELBO")
    while idx < N:
        end = min(idx + batch_size, N)
        # Build batch manually from Subset
        batch_items = [val_data[i] for i in range(idx, end)]
        input_ids = torch.stack([b["labels"] for b in batch_items]).to(device)
        prompt_mask = torch.stack([b["prompt_mask"] for b in batch_items]).to(device)
        B, L = input_ids.shape

        # Save RNG state before sampling so we can retry on OOM
        rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(device) if device != "cpu" else None

        # Sample t ~ U(eps, 1) per sample
        t = torch.rand(B, 1, device=device) * (1.0 - eps) + eps

        # Mask non-prompt tokens independently with prob t
        rand_mask = (torch.rand(B, L, device=device) < t) & ~prompt_mask
        masked_input = torch.where(rand_mask, mask_id, input_ids)

        logits = _try_forward(model, masked_input)
        if logits is None:
            # OOM: halve batch_size permanently and retry from same idx
            torch.cuda.empty_cache()
            if batch_size <= 1:
                raise RuntimeError("OOM with batch_size=1, cannot proceed.")
            batch_size = max(1, batch_size // 2)
            tqdm.write(f"  OOM — halving batch_size to {batch_size}")
            # Restore RNG state so retry produces identical t/masks for this idx
            torch.random.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device)
            continue

        # NLL at ground truth tokens
        if arm_init:
            # logits[:, i] predicts input_ids[:, i+1]
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            nll = -log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            # Align mask: rand_mask[:, 1:] corresponds to positions predicted by logits[:, :-1]
            aligned_mask = rand_mask[:, 1:].float()
            aligned_prompt = prompt_mask[:, 1:]
        else:
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
            aligned_mask = rand_mask.float()
            aligned_prompt = prompt_mask

        # ELBO: (1/t) * sum NLL at masked positions per sample
        masked_nll = nll * aligned_mask
        elbo_per_sample = (1.0 / t.squeeze(1)) * masked_nll.sum(dim=1)

        n_answer = (~aligned_prompt).sum(dim=1).float()

        total_elbo += elbo_per_sample.sum().item()
        total_answer_tokens += n_answer.sum().item()
        n_samples += B

        pbar.update(B)
        idx = end

    pbar.close()
    elbo_per_token = total_elbo / total_answer_tokens if total_answer_tokens > 0 else float("nan")
    return elbo_per_token, n_samples, batch_size


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compute MDM ELBO across checkpoints")
    ap.add_argument("--ckpt_dir", required=True,
                    help="Checkpoint directory (one training run)")
    ap.add_argument("--max_step", type=int, required=True,
                    help="Only evaluate checkpoints <= this step")
    ap.add_argument("--skip", type=int, default=40000,
                    help="Evaluate every N steps (always include first & last)")
    ap.add_argument("--label", default="",
                    help="Display label; if empty, infer from checkpoint config")
    ap.add_argument("--out_dir",
                    default=os.path.join(_HERE, "results"),
                    help="Where to save the output .txt")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (reset per checkpoint for fair comparison)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # ── scan checkpoints ──
    ckpts = scan_checkpoints(args.ckpt_dir, args.max_step, args.skip)
    if not ckpts:
        print("No ema checkpoints found.")
        return
    print(f"Found {len(ckpts)} checkpoints: {[s for s, _ in ckpts]}")

    # ── load config from first checkpoint ──
    first_ckpt = torch.load(ckpts[0][1], map_location="cpu")
    cfg_dict = first_ckpt["config"]

    label = args.label or cfg_dict.get("wandb", {}).get("name", "unknown")

    # ── create model (reused across checkpoints) ──
    model_cfg = dict(cfg_dict["model"])
    arm_init = model_cfg.get("arm_init", "none") != "none"
    mc = MDMConfig(**model_cfg)
    if arm_init:
        mc.predict_next_token = True
    model = MDMTransformer(mc).to(device)
    model.eval()

    # Load first checkpoint weights
    model.load_state_dict(first_ckpt["model_state_dict"], strict=True)
    del first_ckpt

    # ── load val data ──
    data_dir = cfg_dict["data"]["data_dir"]
    val_ratio = cfg_dict["data"].get("val_ratio", 0.02)
    data_seed = cfg_dict["data"].get("seed", 2026)
    mask_id = cfg_dict["data"]["mask_id"]

    _, val_data = split_tinygsm(data_dir, val_ratio=val_ratio, seed=data_seed)

    print(f"Label: {label}")
    print(f"Val samples: {len(val_data)}")
    print(f"mask_id: {mask_id}, ARM init: {arm_init}")
    print(f"Device: {device}")

    # ── write header to output file ──
    out_path = os.path.join(args.out_dir, f"elbo_{label}.txt")
    with open(out_path, "w") as f:
        f.write(f"label: {label}\n")
        f.write(f"ckpt_dir: {args.ckpt_dir}\n")
        f.write(f"n_val_samples: {len(val_data)}\n")
        f.write(f"seed: {args.seed}\n\n")
        f.write(f"{'step':<12}{'elbo_per_token':<16}\n")
        f.write("-" * 28 + "\n")

    # ── evaluate each checkpoint ──
    results = []
    batch_size = args.batch_size

    for step, ckpt_path in ckpts:
        print(f"\n{'='*60}")
        print(f"Step {step}  ({ckpt_path})")
        print(f"{'='*60}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        del ckpt
        torch.cuda.empty_cache()

        elbo, n, batch_size = compute_elbo(
            model, val_data, mask_id, device, batch_size, arm_init, args.seed,
        )
        results.append((step, elbo))
        print(f"  ELBO/token: {elbo:.6f}  (n={n}, batch_size={batch_size})")

        # Append result immediately so partial results survive crashes
        with open(out_path, "a") as f:
            f.write(f"{step:<12}{elbo:<16.6f}\n")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

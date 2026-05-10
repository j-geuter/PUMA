#!/usr/bin/env python3
"""
eval_gsm8k.py

Standalone GSM8K evaluation for a single checkpoint.
Sampling variants are specified via parameter lists:

  --conf_thresholds   confidence-collapse thresholds (default: none)
  --top_k_vals        greedy top-K unmasking, T=0  (default: 10 20)
  --temps_pick        pick positions first (by confidence), sample token with Gumbel at T
                      (default: 0.2 0.5 0.7 0.8 1.0)
  --temps_sample      sample token at every masked position first, keep top-K by prob
                      (default: 0.2 0.5 0.7 0.8 1.0)

Usage:
    python eval_gsm8k.py --ckpt_path /path/to/step=900000.pt
    python eval_gsm8k.py --ckpt_path /path/to/step=900000.pt \\
        --conf_thresholds 0.9 0.95 --top_k_vals 5 10 \\
        --temps_pick 0.5 0.8 --temps_sample 0.5 0.8
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import argparse
import gc
import json
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from model.transformer import MDMTransformer, MDMConfig
from eval.gsm8k_eval import evaluate_samples, get_tokenizer, MASK_ID
from sampling import mdm_sampling, mdm_sampling_block, arm_sampling


def _try_sample(run_fn, batch_X):
    """
    Isolated OOM frame: runs run_fn(batch_X) inside its own stack frame so that
    on OutOfMemoryError, the exception traceback (which holds references to all
    mid-loop tensors in the sampling function) is destroyed when this function
    returns None, allowing empty_cache() to actually free the memory.
    """
    try:
        with torch.no_grad():
            return run_fn(batch_X)
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        gc.collect()


def evaluate_adaptive(model, cfg, device, sampling, initial_batch_size=32):
    """
    Evaluate on GSM8K test set with adaptive batch size: on OOM, halve and retry
    the current batch. Applies per-batch throughout the full eval loop.

    Returns (accuracy, avg_steps_per_sample, avg_tokens_per_step).
    avg_* are None if stats are not available (arm / block strategies).
    """
    mask_id = cfg.data.mask_id
    gsm8k_test_path = "data/gsm8k_test/test_mdm.json"
    with open(gsm8k_test_path, "r") as f:
        records = json.load(f)
    X = np.array([r["input_ids"] for r in records], dtype=np.int64)
    answers = [r["answer"] for r in records]
    N = len(X)

    tokenizer = get_tokenizer()
    arm_init = cfg.model.arm_init != "none"
    use_mdm = cfg.training.strategy not in ("block", "arm")

    def run_sampling(batch_X):
        if cfg.training.strategy == "block":
            return mdm_sampling_block(model, batch_X, cfg.training.block_size, mask_id, sampling, device), None
        elif cfg.training.strategy == "arm":
            return arm_sampling(model, batch_X, mask_id, sampling, device), None
        else:
            xt, stats = mdm_sampling(model, batch_X, mask_id, sampling, device, arm_init=arm_init, return_stats=True)
            return xt, stats

    correct, total = 0, 0
    stat_total_tokens = 0
    stat_total_active_steps = 0
    batch_size = initial_batch_size
    i = 0

    pbar = tqdm(total=N, desc="Evaluating")
    while i < N:
        j = min(i + batch_size, N)
        batch_X = torch.from_numpy(X[i:j]).long().to(device)
        batch_answers = answers[i:j]

        while True:
            result = _try_sample(run_sampling, batch_X)
            if result is not None:
                samples_tensor, batch_stats = result
                break
            # OOM: frame is now gone, safe to free cache
            torch.cuda.empty_cache()
            if batch_size <= 1:
                raise RuntimeError("OOM with batch_size=1, cannot proceed.")
            batch_size = max(1, batch_size // 2)
            j = min(i + batch_size, N)
            batch_X = torch.from_numpy(X[i:j]).long().to(device)
            batch_answers = answers[i:j]
            tqdm.write(f"  OOM — halving batch_size to {batch_size}")

        if batch_stats is not None:
            stat_total_tokens += batch_stats["total_tokens"]
            stat_total_active_steps += batch_stats["total_active_steps"]

        samples_tensor = samples_tensor.masked_fill(samples_tensor == mask_id, tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(samples_tensor.cpu().numpy(), skip_special_tokens=True)
        for sample, answer in zip(decoded, batch_answers):
            if evaluate_samples(sample, answer):
                correct += 1
            total += 1

        pbar.update(j - i)
        i = j

    pbar.close()
    accuracy = correct / total
    if use_mdm and stat_total_active_steps > 0:
        avg_steps = stat_total_active_steps / total
        avg_tok_per_step = stat_total_tokens / stat_total_active_steps
    else:
        avg_steps = avg_tok_per_step = None
    return accuracy, avg_steps, avg_tok_per_step


def build_runs(conf_thresholds, top_k_vals, temps_pick, temps_sample):
    """
    Returns list of (label, description, OmegaConf sampling_cfg) in order:
      1. confidence collapse
      2. greedy top-K
      3. pick-then-sample (existing gumbel)
      4. sample-then-pick (new)
    """
    runs = []

    for tau in conf_thresholds:
        label = f"conf_collapse_t{tau}"
        desc  = f"confidence collapse (threshold={tau}, fallback top-2)"
        cfg   = OmegaConf.create({
            "temperature": 0.0,
            "confidence": "top_k",
            "unmasking_num": 2,
            "confidence_collapse": True,
            "confidence_threshold": float(tau),
            "sample_then_pick": False,
        })
        runs.append((label, desc, cfg))

    for K in top_k_vals:
        label = f"top_k_K{K}"
        desc  = f"greedy top-{K} unmasking (T=0.0)"
        cfg   = OmegaConf.create({
            "temperature": 0.0,
            "confidence": "top_k",
            "unmasking_num": int(K),
            "confidence_collapse": False,
            "sample_then_pick": False,
        })
        runs.append((label, desc, cfg))

    for T in temps_pick:
        label = f"pick_sample_T{T}"
        desc  = f"pick top-2 by confidence, sample token at T={T}"
        cfg   = OmegaConf.create({
            "temperature": float(T),
            "confidence": "top_k",
            "unmasking_num": 2,
            "confidence_collapse": False,
            "sample_then_pick": False,
        })
        runs.append((label, desc, cfg))

    for T in temps_sample:
        label = f"sample_pick_T{T}"
        desc  = f"sample token at T={T}, keep top-2 by sampled prob"
        cfg   = OmegaConf.create({
            "temperature": float(T),
            "confidence": "top_k",
            "unmasking_num": 2,
            "confidence_collapse": False,
            "sample_then_pick": True,
        })
        runs.append((label, desc, cfg))

    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True, help="Path to step=*.pt checkpoint")
    ap.add_argument("--conf_thresholds", nargs="*", type=float, default=[0.8, 0.9, 0.95],
                    metavar="T",
                    help="Confidence-collapse thresholds (default: none)")
    ap.add_argument("--top_k_vals", nargs="*", type=int, default=[2, 5, 10, 20],
                    metavar="K",
                    help="Greedy top-K unmasking values (default: 10 20)")
    ap.add_argument("--temps_pick", nargs="*", type=float,
                    default=[],
                    metavar="T",
                    help="Temperatures for pick-positions-first sampling (default: none)")
    ap.add_argument("--temps_sample", nargs="*", type=float,
                    default=[0.2, 0.5, 0.7, 0.8, 1.0],
                    metavar="T",
                    help="Temperatures for sample-token-first sampling (default: 0.2 0.5 0.7 0.8 1.0)")
    ap.add_argument("--out_dir",
                    default="/n/holylabs/LABS/dam_lab/Users/jgeuter/MDMPre/jay_mdm_playground/PUMA/results",
                    help="Directory to save results txt")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # load checkpoint
    print(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    cfg_dict = ckpt["config"]

    # reconstruct model
    model_cfg_dict = dict(cfg_dict["model"])
    arm_init_path = model_cfg_dict.get("arm_init", "none")
    model_config = MDMConfig(**model_cfg_dict)
    if arm_init_path != "none":
        model_config.predict_next_token = True
    model = MDMTransformer(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    cfg = OmegaConf.create(cfg_dict)
    wandb_name = cfg_dict.get("wandb", {}).get("name", "unknown")
    ckpt_stem = os.path.basename(args.ckpt_path).replace(".pt", "")

    runs = build_runs(
        args.conf_thresholds,
        args.top_k_vals,
        args.temps_pick,
        args.temps_sample,
    )

    print(f"Model: {wandb_name}  |  checkpoint: {ckpt_stem}  |  device: {device}")
    print(f"Running {len(runs)} eval(s)\n")

    results = {}      # label -> accuracy
    stats   = {}      # label -> (avg_steps, avg_tok_per_step)
    for label, desc, sampling in runs:
        torch.cuda.empty_cache()
        print(f"── {label}: {desc} ──")
        acc, avg_steps, avg_tok = evaluate_adaptive(model, cfg, device, sampling, initial_batch_size=args.batch_size)
        results[label] = acc
        stats[label]   = (avg_steps, avg_tok)
        step_str = f"{avg_steps:.2f}" if avg_steps is not None else "n/a"
        tok_str  = f"{avg_tok:.2f}"   if avg_tok  is not None else "n/a"
        print(f"   accuracy: {acc:.4f} ({acc*100:.2f}%)  avg_steps: {step_str}  avg_tok/step: {tok_str}\n")

    # summary
    print("═" * 90)
    print(f"Results for {wandb_name} / {ckpt_stem}")
    print("═" * 90)
    for label, desc, _ in runs:
        avg_steps, avg_tok = stats[label]
        step_str = f"{avg_steps:.2f}" if avg_steps is not None else "n/a"
        tok_str  = f"{avg_tok:.2f}"   if avg_tok  is not None else "n/a"
        print(f"  {label:<30s}  {results[label]:.4f}  steps={step_str}  tok/step={tok_str}")

    # save to txt
    out_path = os.path.join(args.out_dir, f"gsm8k_results_{wandb_name}_{ckpt_stem}.txt")
    col1, col2 = 30, 52
    with open(out_path, "w") as f:
        f.write(f"checkpoint: {args.ckpt_path}\n")
        f.write(f"model: {wandb_name}\n\n")
        f.write(f"{'label':<{col1}}{'description':<{col2}}{'accuracy':<10}{'avg_steps':<12}avg_tok/step\n")
        f.write("-" * (col1 + col2 + 34) + "\n")
        for label, desc, _ in runs:
            avg_steps, avg_tok = stats[label]
            step_str = f"{avg_steps:.2f}" if avg_steps is not None else "n/a"
            tok_str  = f"{avg_tok:.2f}"   if avg_tok  is not None else "n/a"
            f.write(f"{label:<{col1}}{desc:<{col2}}{results[label]:<10.4f}{step_str:<12}{tok_str}\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

"""
Fetch training curves from W&B.

Usage:
    python fetch_wandb.py --name sudoku-puma-single --metric ema-val_acc_top_k_unmasking_2 --max_runs 3
    python fetch_wandb.py --name sudoku-puma-single --metric ema-val_acc_top_k_unmasking_2 --max_runs 3 --save results/sudoku_puma_single.txt
"""
import argparse
import wandb
import numpy as np

ENTITY  = "jaeyeon_kim-harvard-university"
PROJECT = "mdm-pretraining"

def fetch_runs(name: str, metric: str, max_runs: int = 3):
    api = wandb.Api()
    # Filter runs by display name
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"display_name": name},
        order="-created_at",
    )
    results = []
    for run in runs:
        if len(results) >= max_runs:
            break
        if run.state not in ("finished", "running"):
            continue
        hist = run.scan_history(keys=["_step", metric], page_size=10000)
        steps, vals = [], []
        for row in hist:
            if metric in row and row[metric] is not None:
                steps.append(row["_step"])
                vals.append(row[metric])
        if steps:
            results.append({
                "run_id": run.id,
                "run_name": run.name,
                "steps": np.array(steps),
                "vals": np.array(vals),
            })
            print(f"  Found run {run.id} ({run.name}): {len(steps)} points, latest step={steps[-1]}, latest val={vals[-1]:.4f}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="W&B run display name (exact match)")
    parser.add_argument("--metric", required=True, help="Metric key, e.g. ema-val_acc_top_k_unmasking_2")
    parser.add_argument("--max_runs", type=int, default=3)
    parser.add_argument("--save", type=str, default=None, help="Save mean curve to txt file (step, mean, std)")
    args = parser.parse_args()

    print(f"Fetching runs with name='{args.name}', metric='{args.metric}', max_runs={args.max_runs}")
    results = fetch_runs(args.name, args.metric, args.max_runs)

    if not results:
        print("No runs found.")
        return

    # Align all runs to common steps
    common_steps = results[0]["steps"]
    for r in results[1:]:
        common_steps = np.intersect1d(common_steps, r["steps"])

    print(f"\n{len(results)} runs, {len(common_steps)} common steps")

    aligned = []
    for r in results:
        mask = np.isin(r["steps"], common_steps)
        aligned.append(r["vals"][mask])

    aligned = np.array(aligned)  # (num_runs, num_steps)
    mean = aligned.mean(axis=0)
    std = aligned.std(axis=0)

    # Print last few points
    print(f"\nLast 5 common steps:")
    for i in range(-min(5, len(common_steps)), 0):
        print(f"  step {common_steps[i]:>7d}:  mean={mean[i]:.4f}  std={std[i]:.4f}  individual={aligned[:, i]}")

    if args.save:
        import os
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            f.write("# step  mean  std\n")
            for s, m, sd in zip(common_steps, mean, std):
                f.write(f"{s} {m:.6f} {sd:.6f}\n")
        print(f"\nSaved to {args.save}")

if __name__ == "__main__":
    main()

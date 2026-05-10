"""
Plot training curves fetched directly from W&B.

Usage:
    python create_plots.py \
        --runs name1 "[name2,name3]" \
        --metric ema_val_acc_top_k_unmasking_2 \
        --max_runs 3 --ci 2 \
        --legend_title "Inference: Standard Top-K, Data: Sudoku" \
        --dataset sudoku \
        --output results/fig_sudoku \
        --arrow 0 1

Each positional arg to --runs is either a plain run name or a bracketed
comma-separated group.  Plain names are a group of size 1.  For each group
runs are pulled in order (all from the first name, then the second, etc.)
until --max_runs is reached.  The display name is the first name in the
group found in NAME_MAP (falling back to the first name verbatim).
"""

# --- set a private MPL config/cache dir BEFORE importing matplotlib ---
import os, tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import argparse
import numpy as np
import wandb
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, FixedLocator
from matplotlib.lines import Line2D

# -----------------------------
# constants
# -----------------------------
ENTITY  = "jaeyeon_kim-harvard-university"
PROJECT = "mdm-pretraining"

NAME_MAP = {
    "sudoku-baseline-single":              r"MDM",
    "sudoku-puma-single":                  r"MDM + \textbf{PUMA}",
    "sudoku-baseline-single-papl":         r"MDM + PAPL",
    "sudoku-puma-single-papl":             r"MDM + \textbf{PUMA} + PAPL",
    "sudoku-baseline-single-papl-alpha=3": r"MDM + PAPL ($\alpha$=3)",
    "sudoku-baseline-single-papl-alpha=5": r"MDM + PAPL ($\alpha$=5)",
    "sudoku-puma-single-papl-alpha=3":     r"MDM + \textbf{PUMA} + PAPL ($\alpha$=3)",
    "sudoku-puma-single-papl-alpha=5":     r"MDM + \textbf{PUMA} + PAPL ($\alpha$=5)",
    "sudoku-baseline":                     r"MDM",
    "sudoku-puma":                         r"MDM + \textbf{PUMA}",
    "sudoku-puma-no-cc":                   r"MDM + \textbf{PUMA} (no CC)",
    "sudoku-puma-rand10":                  r"MDM + \textbf{PUMA} + 10\% rand",
    "sudoku-puma-rand20":                  r"MDM + \textbf{PUMA} + 20\% rand",
    "sudoku-puma-rand50":                  r"MDM + \textbf{PUMA} + 50\% rand",
    "sudoku-puma-rand100":                 r"MDM + \textbf{PUMA} + 100\% rand",
    "sudoku-puma-single-rand10":           r"MDM + \textbf{PUMA} + 10\% rand",
    "sudoku-puma-single-rand20":           r"MDM + \textbf{PUMA} + 20\% rand",
    "sudoku-puma-warmup15":                r"MDM + \textbf{PUMA} + warmup",
    "sudoku-puma-warmup15-single":         r"MDM + \textbf{PUMA} + warmup",
    "tinygsm-baseline":                    r"MDM",
    "tinygsm-puma":                        r"MDM + \textbf{PUMA}",
    "tinygsm-baseline-papl":              r"MDM + PAPL",
    "tinygsm-puma-papl":                  r"MDM + \textbf{PUMA} + PAPL",
    "tinygsm-puma-arm-init":              r"MDM + \textbf{PUMA} (ARM init)",
    "tinygsm-baseline-arm-init":          r"MDM (ARM init)",
}

# -----------------------------
# styling
# -----------------------------
AX_LABEL_FS = 12
TICK_FS     = 11
LEGEND_FS   = 9
USE_TEX     = True

mpl.rcParams.update({
    "axes.labelsize": AX_LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "axes.linewidth": 1.0,
})
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]

# -----------------------------
# helpers
# -----------------------------
def fetch_runs(name: str, metric: str, max_runs: int = 3, skip: int = 0):
    api = wandb.Api()
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"display_name": name},
        order="-created_at",
    )
    results = []
    skipped = 0
    for run in runs:
        if len(results) >= max_runs:
            break
        if run.state == "created":   # never actually started — no data
            continue
        if skipped < skip:
            print(f"  [{name}] skipping run {run.id} (skip={skip})")
            skipped += 1
            continue
        hist = run.scan_history(keys=["_step", metric], page_size=10000)
        steps, vals = [], []
        for row in hist:
            if metric in row and row[metric] is not None:
                steps.append(row["_step"])
                vals.append(row[metric])
        if steps:
            # deduplicate steps (keep last value per step), sort
            step_to_val = {}
            for s, v in zip(steps, vals):
                step_to_val[s] = v
            sorted_steps = np.array(sorted(step_to_val.keys()))
            sorted_vals  = np.array([step_to_val[s] for s in sorted_steps])
            results.append({"steps": sorted_steps, "vals": sorted_vals})
            print(f"  [{name}] run {run.id}: {len(sorted_steps)} pts, "
                  f"steps {sorted_steps[0]}–{sorted_steps[-1]}, "
                  f"first val={sorted_vals[0]:.4f}, last val={sorted_vals[-1]:.4f}, "
                  f"vals[0:3]={sorted_vals[:3].tolist()}")
    return results


def windowed_ema(y, window=9, alpha=0.25):
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.empty_like(y)
    for t in range(n):
        start = max(0, t - window + 1)
        seg = y[start:t+1]
        s = seg[0]
        for v in seg[1:]:
            s = alpha * v + (1 - alpha) * s
        out[t] = s
    return out


def lighten_hex(hex_color: str, amount: float = 0.20) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02X}{g:02X}{b:02X}"


def assign_colors(run_names):
    """Assign colors based on baseline/puma classification."""
    # 12 visually distinct colors — no repeats within either palette
    BASELINE_PALETTE = [
        "#FF0000",  # red
        "#9400D3",  # violet
        "#8B4513",  # saddle brown
        "#556B2F",  # dark olive green
    ]
    PUMA_PALETTE = [
        "#4169E1",  # royal blue
        "#228B22",  # forest green
        "#FF8C00",  # dark orange
        "#00CED1",  # dark turquoise
        "#9467BD",  # medium purple
        "#DC143C",  # crimson
        "#20B2AA",  # light sea green
        "#FF69B4",  # hot pink
        "#4B0082",  # indigo
        "#DAA520",  # goldenrod
    ]
    FALLBACK_CYCLE = PUMA_PALETTE  # used when no baseline/puma split

    baselines = [(i, n) for i, n in enumerate(run_names) if "baseline" in n.lower()]
    pumas     = [(i, n) for i, n in enumerate(run_names) if "puma" in n.lower()]

    colors = [None] * len(run_names)

    if len(baselines) + len(pumas) == len(run_names) and len(baselines) >= 1 and len(pumas) >= 1:
        # sort within each class by name length (shorter = standard)
        baselines.sort(key=lambda x: len(x[1]))
        pumas.sort(key=lambda x: len(x[1]))

        for ci, (idx, _) in enumerate(baselines):
            colors[idx] = BASELINE_PALETTE[ci % len(BASELINE_PALETTE)]
        for ci, (idx, _) in enumerate(pumas):
            colors[idx] = PUMA_PALETTE[ci % len(PUMA_PALETTE)]
    else:
        for i in range(len(run_names)):
            colors[i] = FALLBACK_CYCLE[i % len(FALLBACK_CYCLE)]

    return [lighten_hex(c, 0.20) for c in colors]


def nice_tick_interval(max_step):
    """Pick a nice tick interval that gives ~4-5 ticks."""
    candidates = [1000, 2000, 5000, 10000, 20000, 25000, 50000, 100000, 200000, 500000]
    for c in candidates:
        n_ticks = max_step / c
        if 3 <= n_ticks <= 6:
            return c
    return max(1, max_step // 5)


def parse_run_groups(raw_args):
    """Parse --runs arguments into groups.

    Examples:
        ["name1", "[name2,name3]"]  ->  [["name1"], ["name2", "name3"]]
        ["name1", "[name2,", "name3]"]  ->  [["name1"], ["name2", "name3"]]
    """
    joined = " ".join(raw_args)
    groups = []
    i = 0
    while i < len(joined):
        if joined[i] == "[":
            j = joined.index("]", i)
            inner = joined[i + 1 : j]
            names = [n.strip() for n in inner.split(",") if n.strip()]
            if names:
                groups.append(names)
            i = j + 1
        elif joined[i] in (" ", ","):
            i += 1
        else:
            j = i
            while j < len(joined) and joined[j] not in (" ", ",", "["):
                j += 1
            name = joined[i:j].strip()
            if name:
                groups.append([name])
            i = j
    return groups


def fetch_runs_multi(names, metric, max_runs=3, skip=0):
    """Fetch runs across multiple names in order until max_runs is reached."""
    results = []
    for name in names:
        remaining = max_runs - len(results)
        if remaining <= 0:
            break
        results.extend(fetch_runs(name, metric, remaining, skip=skip))
    return results


def get_display_name(names):
    """Return the display name for a group: first name found in NAME_MAP."""
    for name in names:
        if name in NAME_MAP:
            return NAME_MAP[name]
    return names[0]


def interpolate_crossing(steps, vals, target):
    """Find the (interpolated) step where vals first reaches target."""
    assert len(steps) == len(vals), (
        f"length mismatch: steps={len(steps)}, vals={len(vals)}")
    idx = np.where(vals >= target)[0]
    if len(idx) == 0:
        return None
    k = int(idx[0])
    if k == 0 or vals[k - 1] >= target:
        return float(steps[k])
    # linear interpolation
    frac = (target - vals[k - 1]) / (vals[k] - vals[k - 1])
    return float(steps[k - 1] + frac * (steps[k] - steps[k - 1]))


def process_run_group(names, metric, max_runs, smooth, skip=0, exclude_indices=None):
    """Fetch, align, smooth, and aggregate a group of seed runs.

    Uses the union of all runs' steps so that partial runs don't truncate
    the plot.  At each step, mean/std are computed over whichever runs have
    data there, so CI bands naturally narrow (or disappear) as runs end.
    """
    raw = fetch_runs_multi(names, metric, max_runs, skip=skip)
    if exclude_indices:
        print(f"  Dropping run indices {exclude_indices} from {len(raw)} fetched runs")
        raw = [r for i, r in enumerate(raw) if i not in exclude_indices]
    if not raw:
        print(f"  WARNING: no runs found for {names}")
        return None

    # smooth each run on its own step grid before aggregating
    smoothed = []
    for r in raw:
        v = windowed_ema(r["vals"], window=9, alpha=0.25) if smooth else r["vals"].copy()
        smoothed.append({"steps": r["steps"], "vals": v,
                         "lookup": dict(zip(r["steps"].tolist(), v.tolist()))})

    # union of all steps
    all_steps = np.array(sorted(set().union(*[set(r["steps"].tolist()) for r in raw])))

    if len(all_steps) == 0:
        print(f"  WARNING: no steps for {names}")
        return None

    means, stds, n_at_step = [], [], []
    for s in all_steps:
        vals_here = [r["lookup"][s] for r in smoothed if s in r["lookup"]]
        means.append(float(np.mean(vals_here)))
        stds.append(float(np.std(vals_here)))
        n_at_step.append(len(vals_here))

    n_at_step_arr = np.array(n_at_step)
    means_arr = np.array(means)
    # print mean at ~10 evenly spaced reference steps for cross-run comparison
    ref_indices = np.linspace(0, len(all_steps) - 1, 10, dtype=int)
    print(f"  DIAG [{names}] mean at reference steps:")
    for ri in ref_indices:
        print(f"    step={all_steps[ri]:>8}  mean={means_arr[ri]:.4f}  n={n_at_step_arr[ri]}")
    return {
        "steps": all_steps,
        "mean": means_arr,
        "std": np.array(stds),
        "n_seeds": int(n_at_step_arr.max()),
        "n_at_step": n_at_step_arr,
    }


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot training curves from W&B")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run names or bracketed groups, e.g.: name1 \"[name2,name3]\"")
    parser.add_argument("--metric", required=True,
                        help="W&B metric key, e.g. ema_val_acc_top_k_unmasking_2")
    parser.add_argument("--max_runs", type=int, default=3,
                        help="Max seeds per run name")
    parser.add_argument("--ci", type=int, default=2,
                        help="CI multiplier (2=95%%, 1=68%%). Ignored if max_runs=1")
    parser.add_argument("--max_iter", type=int, default=None,
                        help="Truncate x-axis at this step")
    parser.add_argument("--arrow", nargs=2, type=int, default=None,
                        metavar=("IDX1", "IDX2"),
                        help="Indices of two runs for speedup arrow")
    parser.add_argument("--arrow_accuracy", type=float, default=None,
                        help="Override arrow y-level")
    parser.add_argument("--legend_title", default="",
                        help="Title line in the legend (omit for no title)")
    parser.add_argument("--plot_title", default=None,
                        help="Title printed below the plot")
    parser.add_argument("--dataset", default="sudoku",
                        choices=["sudoku", "tinygsm"],
                        help="Controls y-axis ticks")
    parser.add_argument("--output", default="results/fig_output",
                        help="Output prefix (saves .png + .pdf)")
    parser.add_argument("--no_smooth", action="store_true",
                        help="Skip windowed EMA smoothing")
    parser.add_argument("--skip_runs", nargs="*", type=int, default=[],
                        metavar="N",
                        help="Per-group number of most-recent runs to skip before "
                             "collecting. One integer per group, in the same order as "
                             "--runs. Unspecified groups default to 0. "
                             "E.g. --skip_runs 1 0 0 0 skips the newest run of group 0.")
    parser.add_argument("--labels", nargs="*", default=[],
                        help="Override display names per group (in same order as --runs). "
                             "Unspecified groups fall back to NAME_MAP / first run name.")
    parser.add_argument("--legend_loc", default="lower right",
                        help="Matplotlib legend location string, e.g. 'upper left', 'lower right'")
    parser.add_argument("--legend_below", action="store_true",
                        help="Place legend below the plot in 2 columns instead of inside it")
    parser.add_argument("--drop_run", nargs="*", default=[],
                        metavar="G:R",
                        help="Drop specific fetched runs: G:R where G=group index, "
                             "R=run index (0-based). E.g. --drop_run 3:2 drops the "
                             "3rd fetched run in group 3. max_runs is auto-increased "
                             "for affected groups.")
    parser.add_argument("--figsize", nargs=2, type=float, default=[4.25, 2.75],
                        metavar=("W", "H"),
                        help="Figure size in inches (default: 4.25 2.75)")
    args = parser.parse_args()

    # ---- parse run groups ----
    groups = parse_run_groups(args.runs)

    # ---- parse --drop_run into per-group sets ----
    drops_per_group = {}
    for dr in args.drop_run:
        g, r = map(int, dr.split(":"))
        drops_per_group.setdefault(g, set()).add(r)

    # ---- fetch data ----
    print(f"Fetching data for {len(groups)} run groups...")
    data = []
    color_keys = []  # first name per group, used for color assignment
    for gi, group_names in enumerate(groups):
        skip = args.skip_runs[gi] if gi < len(args.skip_runs) else 0
        display_name = args.labels[gi] if gi < len(args.labels) else get_display_name(group_names)
        group_drops = drops_per_group.get(gi)
        effective_max = args.max_runs + len(group_drops) if group_drops else args.max_runs
        print(f"  Fetching group {group_names} (display: '{display_name}', skip={skip}"
              f"{f', drops={group_drops}' if group_drops else ''})...")
        d = process_run_group(group_names, args.metric, effective_max, not args.no_smooth,
                              skip=skip, exclude_indices=group_drops)
        if d is None:
            print(f"  Skipping {group_names} (no data)")
            continue
        data.append((display_name, d))
        color_keys.append(group_names[0])

    if not data:
        print("No data to plot.")
        return

    # ---- truncate ----
    if args.max_iter is not None:
        for i, (name, d) in enumerate(data):
            mask = d["steps"] <= args.max_iter
            truncated = {
                "steps": d["steps"][mask],
                "mean": d["mean"][mask],
                "std": d["std"][mask],
                "n_seeds": d["n_seeds"],
            }
            if "n_at_step" in d:
                truncated["n_at_step"] = d["n_at_step"][mask]
            data[i] = (name, truncated)

    # ---- determine arrow target_y ----
    arrow_target_y = None
    step_fast = None

    if args.arrow is not None:
        idx_a, idx_b = args.arrow
        if idx_a >= len(data) or idx_b >= len(data):
            print(f"  WARNING: arrow indices {args.arrow} out of range, skipping arrow")
            args.arrow = None
        else:
            _, d_a = data[idx_a]
            _, d_b = data[idx_b]
            max_a = d_a["mean"].max()
            max_b = d_b["mean"].max()

            if args.arrow_accuracy is not None:
                arrow_target_y = args.arrow_accuracy
            else:
                # worse run = lower max; arrow at its max
                arrow_target_y = min(max_a, max_b)
            print(f"  DIAG arrow_target_y={arrow_target_y:.6f}  (max_a={max_a:.6f}, max_b={max_b:.6f})")
            print(f"  DIAG a steps range: {d_a['steps'][0]}–{d_a['steps'][-1]}, len={len(d_a['steps'])}")
            print(f"  DIAG b steps range: {d_b['steps'][0]}–{d_b['steps'][-1]}, len={len(d_b['steps'])}")
            # show the step where each run's mean is highest
            print(f"  DIAG a peak at step {d_a['steps'][d_a['mean'].argmax()]} val={d_a['mean'].max():.6f}")
            print(f"  DIAG b peak at step {d_b['steps'][d_b['mean'].argmax()]} val={d_b['mean'].max():.6f}")

            # find crossing steps
            cross_a = interpolate_crossing(d_a["steps"], d_a["mean"], arrow_target_y)
            cross_b = interpolate_crossing(d_b["steps"], d_b["mean"], arrow_target_y)
            print(f"  DIAG cross_a={cross_a}, cross_b={cross_b}")

            if cross_a is not None and cross_b is not None and cross_a != cross_b:
                if cross_a < cross_b:
                    step_fast, step_slow = cross_a, cross_b
                else:
                    step_fast, step_slow = cross_b, cross_a
                speedup = step_slow / step_fast if step_fast > 0 else float("nan")
                print(f"  DIAG step_fast={step_fast}, step_slow={step_slow}, speedup={speedup:.4f}x")
            else:
                args.arrow = None  # can't draw arrow

    # ---- colors ----
    colors = assign_colors(color_keys)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=args.figsize, dpi=300)

    for i, (name, d) in enumerate(data):
        legend_name = name  # already resolved by get_display_name
        color = colors[i]
        steps = d["steps"]
        mean  = d["mean"]
        std   = d["std"]
        n_seeds = d["n_seeds"]

        # CI only where >=2 runs contributed (std is meaningful)
        n_at_step = d.get("n_at_step", np.full(len(steps), n_seeds))
        ci_mask = n_at_step >= 2

        if arrow_target_y is not None:
            # faded only where BOTH above target AND to the right of the arrow
            right_of_arrow = (steps >= step_fast) if step_fast is not None else np.ones(len(steps), dtype=bool)
            fade_region = right_of_arrow & (mean > arrow_target_y)
            # two-pass: faded base line everywhere, solid where not faded
            ax.plot(steps, mean, color=color, lw=2.0, alpha=0.35, label=legend_name)
            y_strong = np.where(~fade_region, mean, np.nan)
            ax.plot(steps, y_strong, color=color, lw=2.2, alpha=0.95)

            if n_seeds > 1 and args.ci > 0:
                # CI in non-faded region: full alpha
                strong_ci = ci_mask & ~fade_region
                lo = mean - args.ci * std
                hi = mean + args.ci * std
                ax.fill_between(steps, lo, hi, where=strong_ci, color=color, alpha=0.15)
                # CI in faded region: reduced alpha
                faded_ci = ci_mask & fade_region
                if faded_ci.any():
                    ax.fill_between(steps, lo, hi, where=faded_ci, color=color, alpha=0.05)
        else:
            ax.plot(steps, mean, color=color, lw=2.0, label=legend_name)
            if n_seeds > 1 and args.ci > 0:
                ax.fill_between(steps, mean - args.ci * std, mean + args.ci * std,
                                where=ci_mask, color=color, alpha=0.15)

    # ---- y-axis ----
    if args.dataset == "sudoku":
        pct_ticks = [0.0, 0.20, 0.40, 0.60, 0.80, 1.00]
    else:
        max_val = max(d["mean"].max() for _, d in data)
        # Round up to nearest 5% and add a 5% margin so curves don't touch the top
        y_max = (np.ceil(max_val * 20) + 1) / 20
        tick_step = 0.05 if y_max <= 0.35 else 0.10
        pct_ticks = list(np.round(np.arange(0.0, y_max + 1e-9, tick_step), 10))
        if abs(pct_ticks[-1] - y_max) > 1e-9:
            pct_ticks.append(round(y_max, 10))

    ax.set_ylim(pct_ticks[0], pct_ticks[-1])
    ax.yaxis.set_major_locator(FixedLocator(pct_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, p: rf"{{{int(round(v * 100))}\%}}"
    ))

    # ---- x-axis ----
    max_step = max(d["steps"][-1] for _, d in data)
    tick_int = nice_tick_interval(max_step)
    wanted_steps = np.arange(tick_int, max_step + 1, tick_int)
    ax.set_xticks(wanted_steps)
    ax.set_xticklabels([f"{s // 1000}k" for s in wanted_steps])

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("serif")
        lbl.set_fontweight("bold")

    ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.9, alpha=0.65)
    ax.set_axisbelow(True)

    # ---- legend ----
    legend_handles = []
    if args.legend_title:
        legend_handles.append(Line2D([0], [0], color="none", lw=0, label=args.legend_title))
    for i, (name, _) in enumerate(data):
        legend_handles.append(
            Line2D([0], [0], color=colors[i], lw=3.2,
                   label=name)
        )
    if args.legend_below:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
                  frameon=True, framealpha=0.9, edgecolor="0.8",
                  handlelength=2.0, fontsize=LEGEND_FS, handles=legend_handles)
    else:
        ax.legend(loc=args.legend_loc, frameon=True, framealpha=0.9,
                  edgecolor="0.8", handlelength=2.0, fontsize=LEGEND_FS,
                  handles=legend_handles)

    # ---- speedup arrow ----
    if args.arrow is not None and arrow_target_y is not None:
        ax.annotate(
            "",
            xy=(step_fast, arrow_target_y),
            xytext=(step_slow, arrow_target_y),
            arrowprops=dict(arrowstyle="<->", lw=1.6, color="0.10"),
            zorder=4,
        )
        speedup_text = rf"\textbf{{{speedup:.1f}x faster}}"
        ax.text(
            (step_fast + step_slow) / 2.0, arrow_target_y,
            speedup_text,
            ha="center", va="bottom",
            fontsize=14, fontfamily="Times New Roman", fontweight="bold",
            zorder=5,
        )

    # ---- optional title at bottom ----
    if args.plot_title:
        fig.text(0.5, -0.02, args.plot_title, ha="center", fontsize=11,
                 fontfamily="serif", fontweight="bold")

    # ---- final accuracy summary ----
    print("\n--- Final Accuracy ---")
    for display_name, d in data:
        final_mean = d["mean"][-1]
        final_std = d["std"][-1]
        n_at_end = int(d.get("n_at_step", np.array([d["n_seeds"]]))[-1])
        print(f"  {display_name}: {final_mean*100:.2f}% +/- {final_std*100:.2f}% "
              f"(n={n_at_end}/{d['n_seeds']} seeds at final step)")

    # ---- save ----
    fig.tight_layout()
    plt.ioff()

    out_png = f"{args.output}.png"
    out_pdf = f"{args.output}.pdf"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"\nSaved {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()

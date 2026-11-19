"""Plot learning curves (mean +/- std over seeds) from the per-episode CSVs.

Example:
    python scripts/plot_results.py --algos a2c ppo --out results/a2c_vs_ppo.png

Reads results/runs/<algo>_seed<S>.csv (columns: steps, return).
The curve at step s is the mean return of the last 20 episodes finished by step s,
the same quantity the training log prints as "return(last 20)".
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLOURS = {"a2c": "#2a78d6", "ppo": "#eb6834"}  # blue, orange (colour-blind safe pair)
TARGET = 475
WINDOW = 20


def load_run(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps, returns = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["steps"]))
            returns.append(float(row["return"]))
    return np.array(steps), np.array(returns)


def last_n_curve(steps: np.ndarray, returns: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Mean of the last WINDOW episodes finished at or before each grid step."""
    curve = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        n = np.searchsorted(steps, g, side="right")  # episodes finished by step g
        if n > 0:
            curve[i] = returns[max(0, n - WINDOW):n].mean()
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--algos", nargs="+", default=["a2c", "ppo"])
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--out", default="results/a2c_vs_ppo.png")
    parser.add_argument("--title", default="CartPole-v1")
    args = parser.parse_args()

    grid = np.arange(5_000, args.total_steps + 1, 5_000)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1]})
    finals = {}

    for algo in args.algos:
        files = sorted(Path(args.runs_dir).glob(f"{algo}_seed*.csv"))
        if not files:
            print(f"No CSV files for {algo} in {args.runs_dir}, skipping.")
            continue
        colour = COLOURS.get(algo, "#444444")
        curves, run_finals = [], []
        for p in files:
            s, r = load_run(p)
            curves.append(last_n_curve(s, r, grid))
            run_finals.append(r[-WINDOW:].mean())  # last 20 episodes of the whole run
        curves = np.array(curves)
        mean, std = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)

        for c in curves:  # thin per-seed lines
            ax.plot(grid, c, color=colour, alpha=0.2, linewidth=0.8)
        ax.fill_between(grid, mean - std, np.minimum(mean + std, 500), color=colour, alpha=0.15, linewidth=0)
        ax.plot(grid, mean, color=colour, linewidth=2, label=f"{algo.upper()} (mean of {len(files)} seeds)")
        finals[algo] = run_finals

    ax.axhline(TARGET, color="#52514e", linestyle="--", linewidth=1)
    ax.text(args.total_steps, TARGET + 6, f"{TARGET}", ha="right", va="bottom", fontsize=8, color="#52514e")
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"mean return, last {WINDOW} episodes")
    ax.set_ylim(0, 520)
    ax.set_title(f"{args.title}: mean ± std over seeds (thin lines: single seeds)", fontsize=10, loc="left")
    ax.legend(frameon=False, loc="lower right")

    # Right panel: final value for each seed
    for i, (algo, vals) in enumerate(finals.items()):
        colour = COLOURS.get(algo, "#444444")
        xs = np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals))
        ax2.scatter(xs, vals, color=colour, s=28, zorder=3)
        ax2.hlines(np.mean(vals), i - 0.25, i + 0.25, color=colour, linewidth=2)
    ax2.axhline(TARGET, color="#52514e", linestyle="--", linewidth=1)
    ax2.set_xticks(range(len(finals)))
    ax2.set_xticklabels([a.upper() for a in finals])
    ax2.set_xlim(-0.6, len(finals) - 0.4)
    ax2.set_ylim(0, 520)
    ax2.set_title("final, per seed (bar = mean)", fontsize=10, loc="left")

    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e5e4e0", linewidth=0.6)
        a.set_axisbelow(True)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

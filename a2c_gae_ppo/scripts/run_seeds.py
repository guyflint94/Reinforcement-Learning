"""Run train.py for several seeds, one after another.

Example:
    python scripts/run_seeds.py --algo ppo --lr 0.0003 --seeds 0 1 2 3 4
    python scripts/run_seeds.py --algo a2c --lr 0.0007 --seeds 0 1 2 3 4

Each run writes results/runs/<algo>_seed<S>.csv (per-episode returns) and
results/runs/<algo>_seed<S>.txt (the printed training log).
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["a2c", "ppo"], required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--out-dir", type=str, default="results/runs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        cmd = [sys.executable, "-m", "actor_critic_gae.train",
               "--algo", args.algo, "--lr", str(args.lr), "--seed", str(seed),
               "--total-steps", str(args.total_steps), "--out-dir", str(out_dir)]
        print(f"=== {args.algo} seed {seed} ===", flush=True)
        log_file = out_dir / f"{args.algo}_seed{seed}.txt"
        with open(log_file, "w") as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:  # echo to the terminal and save to the file
                print(line, end="")
                f.write(line)
            proc.wait()
        if proc.returncode != 0:
            print(f"Run failed (seed {seed}), see {log_file}")
            break


if __name__ == "__main__":
    main()

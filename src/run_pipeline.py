import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = sys.executable

def run_cmd(cmd, desc):
    print(f"\n{'='*70}\n[STEP] {desc}\n  CMD: {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    t1 = time.time()
    print(f"[*] Completed in {t1 - t0:.1f}s\n", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seat-turns", type=int, default=75000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cls", type=str, default="micro")
    args = parser.parse_args()

    data_file = PROJECT_ROOT / "data" / "tactical_teacher.jsonl.gz"
    runs_dir = PROJECT_ROOT / "runs" / f"tactical_{args.cls}"
    out_dir = PROJECT_ROOT / "models" / f"tactical_{args.cls}"

    # Step 1: Collect
    run_cmd(
        [PYTHON, "src/collect_tactical_data.py", "--seat-turns", str(args.seat_turns), "--out", str(data_file)],
        f"Collecting {args.seat_turns:,} tactical seat-turns"
    )

    # Step 2: Train
    run_cmd(
        [PYTHON, "-m", "tb_baselines.train.bc", "--class", args.cls, "--epochs", str(args.epochs),
         "--seed", str(args.seed), "--data", str(data_file), "--out", str(runs_dir)],
        f"Training {args.cls} model for {args.epochs} epochs"
    )

    # Read training history
    history_file = runs_dir / "history.json"
    with open(history_file) as f:
        history = json.load(f)
    last_epoch = history["epochs"][-1]
    val_acc = last_epoch["val_acc"]
    method_str = f"Tactical BC: {args.epochs} epochs, {args.seat_turns:,} seat-turns, {val_acc:.1%} val_acc, seed {args.seed}"
    print(f"[+] Training achieved {val_acc:.2%} validation agreement!", flush=True)

    # Step 3: Export
    run_cmd(
        [PYTHON, "-m", "tb_baselines.export", "--class", args.cls, "--weights", str(runs_dir / "best.pt"),
         "--out", str(out_dir), "--method", method_str],
        f"Exporting model to ONNX & generating manifest"
    )

    # Copy exported files to root
    for fname in ["model.onnx", "manifest.json", "metrics.json", "card.md"]:
        shutil.copyfile(out_dir / fname, PROJECT_ROOT / fname)
    print(f"\n[+] Successfully updated root submission files (model.onnx, manifest.json, metrics.json, card.md)", flush=True)

    # Step 4: Verify with tinybrains check
    run_cmd(
        ["tinybrains", "check", "model.onnx", "manifest.json"],
        "Verifying model with official tinybrains admission gate"
    )

    # Step 5: Run Evaluation Tournament
    run_cmd(
        ["tinybrains", "matches/eval_2p.json", "-v"],
        "Tournament 1: 2-Player Duels vs Platform Baselines"
    )

    run_cmd(
        ["tinybrains", "matches/eval_4p.json", "-v"],
        "Tournament 2: 4-Player FFA Championship"
    )

    print("\n" + "="*70)
    print("  ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*70, flush=True)

if __name__ == "__main__":
    main()

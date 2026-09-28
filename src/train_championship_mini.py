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

PYTHON = "/opt/miniconda3/bin/python3"

def run_cmd(cmd, desc):
    print(f"\n{'='*75}\n[STEP] {desc}\n  CMD: {' '.join(cmd)}\n{'='*75}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    t1 = time.time()
    print(f"[*] Completed in {t1 - t0:.1f}s\n", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seat-turns", type=int, default=120000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_file = PROJECT_ROOT / "data" / "season_tactical_120k.jsonl.gz"
    runs_dir = PROJECT_ROOT / "runs" / "championship_mini"
    out_dir = PROJECT_ROOT / "models" / "championship_mini"

    # Step 1: Collect across all 32 season maps
    run_cmd(
        [PYTHON, "src/collect_tactical_data.py", "--maps", "season_maps", "--seat-turns", str(args.seat_turns),
         "--waves", "4", "--matches-per-wave", "8", "--max-turns", "250", "--out", str(data_file)],
        f"Collecting {args.seat_turns:,} tactical decisions across all 32 season maps"
    )

    # Step 2: Train MiniTrunk (230,149 parameters)
    run_cmd(
        [PYTHON, "-m", "tb_baselines.train.bc", "--class", "mini", "--epochs", str(args.epochs),
         "--seed", str(args.seed), "--data", str(data_file), "--out", str(runs_dir)],
        f"Training MiniTrunk (230,149 params) on Apple Silicon GPU for {args.epochs} epochs"
    )

    # Read training history
    history_file = runs_dir / "history.json"
    with open(history_file) as f:
        history = json.load(f)
    last_epoch = history["epochs"][-1]
    val_acc = last_epoch["val_acc"]
    val_loss = last_epoch["val_loss"]
    method_str = f"Championship Mini-Trunk: {args.epochs} epochs, {args.seat_turns:,} seat-turns across 32 season maps, {val_acc:.1%} val_acc, val_loss {val_loss:.4f}"
    print(f"[+] Training complete: {val_acc:.2%} validation accuracy, loss: {val_loss:.4f}", flush=True)

    # Step 3: Export to ONNX with fp16 initializers
    run_cmd(
        [PYTHON, "-m", "tb_baselines.export", "--class", "mini", "--weights", str(runs_dir / "best.pt"),
         "--out", str(out_dir), "--method", method_str],
        "Exporting MiniTrunk to ONNX & generating manifest"
    )

    # Copy files to root for instant submission
    for fname in ["model.onnx", "manifest.json", "metrics.json", "card.md"]:
        shutil.copyfile(out_dir / fname, PROJECT_ROOT / fname)
    print(f"\n[+] Successfully updated root submission files (model.onnx, manifest.json, metrics.json, card.md)", flush=True)

    # Step 4: Admission verification
    run_cmd(
        ["tinybrains", "check", "model.onnx", "manifest.json"],
        "Verifying model with official tinybrains admission gate"
    )

    # Step 5: Test on real season maps
    run_cmd(
        ["tinybrains", "matches/eval_4p.json", "-v"],
        "Tournament Evaluation: 4-Player FFA Championship"
    )

    print("\n" + "="*75)
    print("  CHAMPIONSHIP MINI MODEL SUCCESSFULLY BUILT AND READY FOR TOP 1!")
    print("="*75, flush=True)

if __name__ == "__main__":
    main()

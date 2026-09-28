import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tb_baselines import nets
from tb_baselines.export import classes
from tb_baselines.train.bc import Rows, batches, device, tensors, ant_logits

def train_and_export_v5(total_epochs=15, resume_from_v2=True, batch_size=32, lr=1e-3, seed=42):
    data_file = PROJECT_ROOT / "data" / "season_tactical_120k.jsonl.gz"
    v2_pt = PROJECT_ROOT / "runs" / "championship_mini" / "best.pt"
    runs_dir = PROJECT_ROOT / "runs" / "v5_mini"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_dir = PROJECT_ROOT / "models" / "v5_mini"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = device()

    print(f"\n{'='*75}\n[STEP 1] Loading Season Dataset (120,000 seat-turns)\n{'='*75}", flush=True)
    t0 = time.time()
    rows = Rows(data_file)
    print(f"[+] Loaded {len(rows):,} rows in {time.time()-t0:.2f}s", flush=True)

    order = list(range(len(rows)))
    rng.shuffle(order)
    cut = int(len(order) * 0.95)
    train_idx, val_idx = order[:cut], order[cut:]
    print(f"[+] Dataset split: {len(train_idx):,} train / {len(val_idx):,} validation turns", flush=True)

    # MiniTrunk(64, 6) — 230,149 parameters
    spec = classes()["mini"]
    model = nets.build(spec).to(dev)
    params_count = nets.policy_params(model)
    print(f"[+] Model Architecture: MiniTrunk(64, 6) — {params_count:,} parameters (device: {dev.type})", flush=True)

    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    if resume_from_v2 and v2_pt.exists():
        print(f"[+] Loading v2 weights ({v2_pt}) [10 epochs pre-trained at 94.6% val_acc, 0.17 loss]", flush=True)
        ckpt = torch.load(v2_pt, map_location=dev)
        model.load_state_dict(ckpt["trunk"])
        start_epoch = 11
        best_val_loss = 0.1722
        print(f"[+] Resuming training from Epoch 11 to Epoch {total_epochs} (target: 15 epochs total)!", flush=True)
    else:
        print(f"[+] Starting training from scratch (Epoch 1 to {total_epochs})", flush=True)

    remaining_epochs = total_epochs - start_epoch + 1
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(remaining_epochs, 1), eta_min=1e-5)
    val_chunks = batches(rows, val_idx, batch_size, rng, shuffle=False)

    best_pt = runs_dir / "best.pt"

    print(f"\n{'='*75}\n[STEP 2] Training MiniTrunk for Epochs {start_epoch} to {total_epochs} with Live Progress\n{'='*75}", flush=True)

    for epoch in range(start_epoch, total_epochs + 1):
        ep_t0 = time.time()
        chunks = batches(rows, train_idx, batch_size, rng)
        total_chunks = len(chunks)
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_ants = 0

        window_loss = 0.0
        window_correct = 0
        window_ants = 0
        window_t0 = time.time()

        for i, batch in enumerate(chunks, 1):
            boards, b_idx, r_idx, c_idx, y = tensors(rows, batch, dev)
            if y.numel() == 0:
                continue

            opt.zero_grad(set_to_none=True)
            logits = ant_logits(model(boards), b_idx, r_idx, c_idx)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            n = y.numel()
            n_corr = int((logits.argmax(1) == y).sum())
            loss_val = loss.detach().item()

            total_loss += loss_val * n
            total_correct += n_corr
            total_ants += n

            window_loss += loss_val * n
            window_correct += n_corr
            window_ants += n

            if i % 50 == 0 or i == total_chunks:
                now = time.time()
                dt = max(now - window_t0, 1e-4)
                speed = 50 / dt
                remaining_chunks = total_chunks - i
                eta_sec = remaining_chunks / max(speed, 0.1)
                cur_acc = window_correct / max(window_ants, 1)
                cur_loss = window_loss / max(window_ants, 1)
                pct = (i / total_chunks) * 100

                print(
                    f"  [Epoch {epoch}/{total_epochs}] Batch {i:4d}/{total_chunks} ({pct:5.1f}%) "
                    f"| Loss: {cur_loss:.4f} | Acc: {cur_acc*100:5.1f}% | {speed:4.1f} b/s | ETA: {int(eta_sec//60):02d}m {int(eta_sec%60):02d}s",
                    flush=True
                )
                window_loss = 0.0
                window_correct = 0
                window_ants = 0
                window_t0 = time.time()

        # Validation at epoch end
        print(f"  [Epoch {epoch}/{total_epochs}] Evaluating on validation set...", flush=True)
        model.eval()
        v_loss = v_corr = v_ants = 0
        with torch.no_grad():
            for batch in val_chunks:
                boards, b_idx, r_idx, c_idx, y = tensors(rows, batch, dev)
                if y.numel() == 0:
                    continue
                logits = ant_logits(model(boards), b_idx, r_idx, c_idx)
                loss = F.cross_entropy(logits, y)
                n = y.numel()
                v_loss += loss.item() * n
                v_corr += int((logits.argmax(1) == y).sum())
                v_ants += n

        val_loss = v_loss / max(v_ants, 1)
        val_acc = v_corr / max(v_ants, 1)
        train_loss = total_loss / max(total_ants, 1)
        train_acc = total_correct / max(total_ants, 1)
        ep_time = time.time() - ep_t0
        sched.step()

        print(
            f"\n  [★ EPOCH {epoch} COMPLETE] Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.1f}%) "
            f"| Val Loss: {val_loss:.4f} (Acc: {val_acc*100:.1f}%) | Time: {ep_time:.1f}s\n",
            flush=True
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "seconds": round(ep_time, 1)
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  [+] New best model! Saving to {best_pt}", flush=True)
            torch.save({
                "trunk": model.state_dict(),
                "class": "mini",
                "engine_digest": rows.header["engine_digest"]
            }, best_pt)
        else:
            # Also keep latest if best wasn't updated
            if not best_pt.exists():
                torch.save({
                    "trunk": model.state_dict(),
                    "class": "mini",
                    "engine_digest": rows.header["engine_digest"]
                }, best_pt)

    (runs_dir / "history.json").write_text(json.dumps({
        "class": "mini", "method": "bc", "arch": spec["arch"], "spec": spec,
        "data": str(data_file), "epochs": history,
        "engine_digest": rows.header["engine_digest"], "device": dev.type
    }, indent=2) + "\n")

    # Step 3: Export to ONNX
    print(f"\n{'='*75}\n[STEP 3] Exporting FatAss v5 to ONNX & Manifest\n{'='*75}", flush=True)
    method_str = f"FatAss v5 Championship: MiniTrunk(64,6), 15 epochs, {val_acc:.1%} val_acc, val_loss {val_loss:.4f}"
    from tb_baselines.export import main as export_main
    export_main(["--class", "mini", "--weights", str(best_pt), "--out", str(out_dir), "--method", method_str])

    # Copy to root
    for fname in ["model.onnx", "manifest.json", "metrics.json", "card.md"]:
        shutil.copyfile(out_dir / fname, PROJECT_ROOT / fname)
    print(f"[+] Root files updated: model.onnx, manifest.json, metrics.json, card.md", flush=True)

    # Step 4: TinyBrains admission check
    print(f"\n{'='*75}\n[STEP 4] Official TinyBrains Admission Verification\n{'='*75}", flush=True)
    import subprocess
    subprocess.run(["tinybrains", "check", "model.onnx", "manifest.json"], check=True)

    # Step 5: Head-to-Head Tournament against v2
    print(f"\n{'='*75}\n[STEP 5] Automated Head-to-Head Benchmark vs FatAss v2\n{'='*75}", flush=True)
    subprocess.run([sys.executable, "src/benchmark_tournament.py"], check=True)

    print(f"\n{'='*75}\n  FATASS V5 FULLY CERTIFIED & READY FOR TOP 1 ON TINYBRAINS.DEV!\n{'='*75}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    train_and_export_v5(total_epochs=args.epochs, resume_from_v2=not args.no_resume)

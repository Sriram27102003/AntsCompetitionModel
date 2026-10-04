"""Behaviour cloning from bit-packed shards (prep_fast.py): MiniTrunk(64,6) with the two memory
planes, the same network and export path as train.bc --memory, without re-parsing JSON each epoch.

    python train_fast.py packed/*.npz --init ../runs/v5_mini/best.pt --epochs 4 --out runs/v10

The checkpoint it writes is the same shape as train.bc's, so tb_baselines.export reads it.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tb_baselines import nets
from tb_baselines.export import classes
from tb_baselines.planes import N_MEMORY, N_PLANES
from tb_baselines.train.bc import device, override

N_IN = N_PLANES + N_MEMORY


def load(paths: list[Path]) -> dict:
    """Shards merged by board size: {(h, w): dict of arrays}."""
    out: dict = {}
    for p in paths:
        z = np.load(p)
        for k in {name.rsplit("_x", 1)[0] for name in z.files if name.endswith("_x")}:
            h, w = map(int, k.split("x"))
            g = out.setdefault((h, w), {"x": [], "row": [], "r": [], "c": [], "y": []})
            base = sum(len(x) for x in g["x"])
            g["x"].append(z[f"{k}_x"])
            g["row"].append(z[f"{k}_ant_row"].astype(np.int64) + base)
            g["r"].append(z[f"{k}_ant_r"])
            g["c"].append(z[f"{k}_ant_c"])
            g["y"].append(z[f"{k}_ant_y"])
    for g in out.values():
        for key in list(g):
            g[key] = np.concatenate(g[key])
        # ants of each row, for gathering a batch's labels
        order = np.argsort(g["row"], kind="stable")
        for key in ("row", "r", "c", "y"):
            g[key] = g[key][order]
        g["start"] = np.searchsorted(g["row"], np.arange(len(g["x"]) + 1))
    return out


def batch(g: dict, rows: np.ndarray, size, dev):
    h, w = size
    bits = np.unpackbits(g["x"][rows], axis=1)[:, : N_IN * h * w]
    x = torch.from_numpy(bits.reshape(len(rows), N_IN, h, w).astype(np.int8))
    idx = np.concatenate([np.arange(g["start"][r], g["start"][r + 1]) for r in rows])
    local = np.repeat(np.arange(len(rows)), np.diff(g["start"])[rows])
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.int64)).to(dev)
    return (x[:, :N_PLANES].to(dev), x[:, N_PLANES:].to(dev), t(local),
            t(g["r"][idx]), t(g["c"][idx]), t(g["y"][idx]))


def fight_weights(board, b, r, c, weight: float):
    """Per-ant loss weights: `weight` for an ant with an enemy in fight range (wrapped d2 <= 20, the
    range in which both sides can be in attack range after moving), 1 otherwise. v10d agreed with
    the teacher on 75% of those decisions against 89% elsewhere, and they are where ants die."""
    foes = board[:, 1:2].float()
    k = torch.tensor([[float(dr * dr + dc * dc <= 20) for dc in range(-4, 5)] for dr in range(-4, 5)],
                     device=board.device).reshape(1, 1, 9, 9)
    near = F.conv2d(F.pad(foes, (4, 4, 4, 4), mode="circular"), k)[:, 0] > 0
    return 1.0 + (weight - 1.0) * near[b, r, c].float()


def run(model, data, chunks, dev, opt=None):
    model.train(opt is not None)
    tot = correct = seen = 0
    for size, rows in chunks:
        board, mem, b, r, c, y = batch(data[size], rows, size, dev)
        if not len(y):
            continue
        with torch.set_grad_enabled(opt is not None):
            policy = model(board, mem)[0]
            logits = policy[b, :, r, c]
            loss = F.cross_entropy(logits, y)
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        tot += loss.item() * len(y)
        correct += int((logits.argmax(1) == y).sum())
        seen += len(y)
    return tot / max(seen, 1), correct / max(seen, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", type=Path)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", type=int, default=64,
                    help="trunk width; wider than --init's and the old function is kept exactly")
    ap.add_argument("--fight-weight", type=float, default=1.0,
                    help="loss weight of an ant with an enemy in fight range (held-out loss stays unweighted)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = device()

    t0 = time.time()
    data = load(a.shards)
    n_rows = sum(len(g["x"]) for g in data.values())
    print(f"{n_rows:,} rows over {len(data)} board sizes, loaded in {time.time() - t0:.0f}s", flush=True)

    class A:
        arch, channels, blocks = "mini_trunk", a.channels, 6
    spec = override(classes()["mini"], A)
    model = nets.build(spec, memory=True)
    if a.init:
        src = torch.load(a.init, map_location="cpu", weights_only=False)["trunk"]
        dst = model.state_dict()
        for k, v in src.items():
            t = dst[k] if k in dst else dst["trunk." + k]
            if t.shape == v.shape:
                t.copy_(v)
            elif v.dim() == 1:                      # a bias: the old channels keep theirs
                t[: v.shape[0]].copy_(v)
            else:
                # A wider or newly fed layer starts as the same function: the old channels keep
                # their weights and read nothing from the new inputs (zeros), while the new
                # channels keep their random incoming weights so they are alive and can be learned.
                o, i = v.shape[:2]
                t[:o, i:].zero_()
                t[:o, :i].copy_(v)
        model.load_state_dict(dst)
        print(f"warm start from {a.init}", flush=True)
    model.to(dev)

    train, val = [], []
    for size, g in data.items():
        idx = np.arange(len(g["x"]))
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - a.holdout))
        train += [(size, idx[i : i + a.batch]) for i in range(0, cut, a.batch)]
        val += [(size, idx[cut:][i : i + a.batch]) for i in range(0, len(idx) - cut, a.batch)]

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(train))
    a.out.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        rng.shuffle(train)
        model.train()
        tot = correct = seen = 0
        for step, (size, rows) in enumerate(train):
            board, mem, b, r, c, y = batch(data[size], rows, size, dev)
            if not len(y):
                continue
            policy = model(board, mem)[0]
            logits = policy[b, :, r, c]
            if a.fight_weight != 1.0:
                w = fight_weights(board, b, r, c, a.fight_weight)
                loss = (F.cross_entropy(logits, y, reduction="none") * w).sum() / w.sum()
            else:
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
            seen += len(y)
            if step % 50 == 0 and dev.type == "mps":
                # MPS keeps a freed buffer for every tensor shape it has seen, and there are 23 board
                # sizes: left alone the process reached 19 GB and pushed the Mac deep into swap.
                torch.mps.empty_cache()
            if step % 500 == 0:
                print(f"    step {step}/{len(train)}  loss {tot / max(seen, 1):.4f}  "
                      f"acc {correct / max(seen, 1):.1%}  {time.time() - t0:.0f}s", flush=True)
        vl, va = run(model, data, val, dev)
        took = time.time() - t0
        print(f"  epoch {epoch}/{a.epochs}  train {tot / seen:.4f} / {correct / seen:.1%}   "
              f"held out {vl:.4f} / {va:.1%}   {took:.0f}s", flush=True)
        history.append({"epoch": epoch, "train_loss": tot / seen, "train_acc": correct / seen,
                        "val_loss": vl, "val_acc": va, "seconds": round(took, 1)})
        if vl < best:
            best = vl
            torch.save({"trunk": model.state_dict(), "class": "mini", "memory": True,
                        "engine_digest": "sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1"},
                       a.out / "best.pt")
    (a.out / "history.json").write_text(json.dumps(
        {"method": "bc (train_fast)", "spec": spec, "memory": True,
         "shards": [str(p) for p in a.shards], "init": str(a.init), "epochs": history}, indent=2))
    print(f"  -> {a.out / 'best.pt'}  (best held-out loss {best:.4f})")


if __name__ == "__main__":
    main()

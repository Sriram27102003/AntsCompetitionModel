"""How do the ladder's best models play, and how far is the strong teacher from them?

Reads rows made by replays_to_dataset.py (each row: what a seat saw, what it ordered, whose seat),
and prints per model: colony size by turn, how often it holds, and how often the strong teacher
would have ordered the same move, split by whether an enemy is in fight range.

    python champion_study.py data/champions.jsonl.gz [--sample 6000]
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import defaultdict

import numpy as np

from strong_teacher import StrongTeacher, wrap_d2

MOVES = "NESW-"
MARKS = (10, 25, 50, 100, 200, 400)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--sample", type=int, default=6000, help="rows the teacher is asked about")
    a = ap.parse_args()
    rng = random.Random(0)
    size = defaultdict(lambda: defaultdict(list))            # model -> turn mark -> colony sizes
    seats = defaultdict(set)
    holds = defaultdict(lambda: [0, 0])
    keep: list[dict] = []
    n = 0
    with gzip.open(a.rows, "rt") as f:
        f.readline()
        for line in f:
            r = json.loads(line)
            n += 1
            who, (match, seat, turn) = r["who"], r["k"]
            seats[who].add((match, seat))
            if turn in MARKS:
                size[who][turn].append(len(r["o"]["mine"]))
            holds[who][0] += r["a"].count("-")
            holds[who][1] += len(r["a"])
            if len(keep) < a.sample:                          # reservoir sample
                keep.append(r)
            elif rng.random() < a.sample / n:
                keep[rng.randrange(a.sample)] = r

    print(f"{n:,} rows\n\ncolony size at turn (median of the seats still alive):")
    print(f"{'model':24s} seats " + " ".join(f"{'t' + str(t):>6s}" for t in MARKS) + "   holds")
    for who in sorted(seats, key=lambda w: -len(seats[w])):
        cells = " ".join(f"{np.median(size[who][t]):6.0f}" if size[who][t] else "     -" for t in MARKS)
        print(f"{who[:24]:24s} {len(seats[who]):5d} {cells}   {100 * holds[who][0] / max(holds[who][1], 1):4.1f}%")

    teacher = StrongTeacher()
    agree = defaultdict(lambda: [0, 0])
    for r in keep:
        o = r["o"]
        mine = np.array(o["mine"], dtype=np.int64).reshape(-1, 2)
        if not len(mine):
            continue
        want = np.asarray(teacher.orders(o, r.get("m") or None))
        got = np.array([MOVES.index(c) if c in MOVES else 4 for c in r["a"]])
        ok = want == got
        foes = np.array([f[:2] for f in o["foes"]], dtype=np.int64).reshape(-1, 2)
        rows, cols = o["size"]
        near = (wrap_d2(mine, foes, rows, cols).min(1) <= 20) if len(foes) else np.zeros(len(mine), bool)
        for key, mask in (("all", np.ones(len(mine), bool)), ("enemy in fight range", near), ("no enemy near", ~near)):
            agree[(r["who"], key)][0] += int(ok[mask].sum())
            agree[(r["who"], key)][1] += int(mask.sum())
    print(f"\nstrong teacher would order the same move ({len(keep):,} sampled rows):")
    for (who, key), (x, t) in sorted(agree.items()):
        print(f"  {who[:24]:24s} {key:22s} {100 * x / max(t, 1):5.1f}%  of {t:,} ant decisions")


if __name__ == "__main__":
    main()

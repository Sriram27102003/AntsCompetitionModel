"""Local round robin on the Hive Minds boards, with the engine the season plays.

    python arena.py opp/v5 opp/nano-bc-max-r --per-size 2 --seeds 1 2

Every pairing plays 2-seat boards in both seat orders; 4- and 6-seat boards alternate the entrants
around the table. Prints each entrant's win rate (sole or shared rank 1) and mean placement.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLI = os.environ.get(
    "TINYBRAINS",
    str(HERE / "bin" / "tinybrains"),
)
LINE = re.compile(r"^(\S+)\s+(\w+)\s+(\d+) turns\s+board (\S+)\s+ranks (.+?)\s+scores")


def seat(i: int, d: Path) -> dict:
    return {"seat": i, "weights": str(d / "model.onnx"), "manifest": str(d / "manifest.json"),
            "label": d.name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("entrants", nargs="+", type=Path)
    ap.add_argument("--maps", type=Path, default=HERE / "maps_hive")
    ap.add_argument("--per-size", type=int, default=2, help="boards per (seat count, size) group")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--max-turns", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=HERE / "arena")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    ents = [e.resolve() for e in a.entrants]

    rng = random.Random(0)
    groups = defaultdict(list)
    for f in sorted(a.maps.glob("*.json")):
        size, _, seats = f.stem.split("-")[:3]
        groups[(int(seats[:-1]), size)].append(f)
    boards = [f for g in groups.values() for f in rng.sample(g, min(a.per_size, len(g)))]

    waves = defaultdict(list)
    for f in boards:
        n = int(f.stem.split("-")[2][:-1])
        for s in a.seeds:
            if n == 2:
                for x, y in itertools.permutations(ents, 2):
                    waves[2].append((f, s, [x, y]))
            else:
                for off in range(len(ents)):
                    waves[n].append((f, s, [ents[(i + off) % len(ents)] for i in range(n)]))

    stats = defaultdict(lambda: {"n": 0, "win": 0, "place": 0.0})
    a.out.mkdir(parents=True, exist_ok=True)
    for n, rows in sorted(waves.items()):
        spec = {"game": "ants", "vars": {"max_turns": a.max_turns}, "rows": [
            {"id": f"{f.stem}-s{s}-{k}", "seed": s, "map": str(f), "seat_count": n,
             "seats": [seat(i, d) for i, d in enumerate(ds)]}
            for k, (f, s, ds) in enumerate(rows)]}
        path = a.out / f"{a.tag}-{n}p.json"
        path.write_text(json.dumps(spec))
        res = subprocess.run([CLI, "run", str(path), "--out", str(a.out / f"{a.tag}-{n}p")],
                             cwd=HERE, capture_output=True, text=True)
        for line in res.stdout.splitlines() + res.stderr.splitlines():
            m = LINE.match(line.strip())
            if not m:
                continue
            ranks = [tuple(x.split("=")) for x in m.group(5).split()]
            seen = set()
            for label, r in ranks:
                r = int(r)
                key = (label, n)
                st = stats[key]
                # an entrant holding several seats counts its best seat once
                if (label) in seen:
                    continue
                seen.add(label)
                best = min(int(rr) for ll, rr in ranks if ll == label)
                st["n"] += 1
                st["win"] += best == 1
                st["place"] += (best - 1) / (n - 1)
        if res.returncode:
            print(res.stderr[-2000:])

    print(f"{'entrant':<20}{'seats':>6}{'games':>7}{'win':>7}{'place':>7}")
    for (label, n), st in sorted(stats.items()):
        print(f"{label:<20}{n:>6}{st['n']:>7}{st['win'] / st['n']:>7.0%}{st['place'] / st['n']:>7.2f}")
    tot = defaultdict(lambda: [0, 0, 0.0])
    for (label, _), st in stats.items():
        t = tot[label]
        t[0] += st["n"]; t[1] += st["win"]; t[2] += st["place"]
    print("overall")
    for label, (g, w, p) in sorted(tot.items(), key=lambda kv: -kv[1][1] / kv[1][0]):
        print(f"  {label:<20} {g:>4} games  win {w / g:.0%}  placement {p / g:.2f} (0 = first)")


if __name__ == "__main__":
    main()

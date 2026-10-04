"""Turn downloaded QA replays into a behaviour-cloning dataset of the leaders' own moves.

A replay stores the board, the seed and every seat's orders, not what a seat saw. So each match is
played again through `tinybrains env` on the same board with the same seed, feeding every seat its
recorded orders; the env hands back each seat's fogged observation, which is exactly what that
model saw on the ladder. A match counts only if the replayed ending matches the recorded one.

Rows use `collect.py`'s format (`o`, `a`, `m`, `k`), with `m` the numpy `remember` state, so
`train.bc --memory` reads them unchanged.

    python replays_to_dataset.py tiger --models TigerAnts FireAnts --out data/leaders.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import tempfile
from collections import Counter
from pathlib import Path

from tb_baselines.env import Env
from tb_baselines.planes import MEMORY, remember


def versions(match: dict) -> dict[str, str]:
    """Version id -> model name, from the match record."""
    return {s["version_id"]: s["model"] for s in match.get("seats") or match["players"]}   # production records say `players`


def replay(path: Path, match: dict, keep: set[str], out, stats: Counter) -> None:
    rp = json.loads(path.read_text())
    names = versions(match)
    seat_model = {s["seat"]: names.get(s["model"].removeprefix("tb.v"), "?") for s in rp["seats"]}
    wanted = {seat for seat, name in seat_model.items() if name in keep or "ALL" in keep}
    if not wanted:
        return
    orders = {d["t"]: d["a"] for d in rp["deltas"]}
    rows: list[str] = []
    state: dict[int, dict | None] = {}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(rp["map"], f)
        board = f.name
    with Env(waves=1, matches_per_wave=1, max_turns=rp["max_turns"], seed=rp["seed"],
             maps=board) as env:
        step = env.reset()
        ended = None
        # The env refills a finished match with a fresh one, so the loop ends on the first ending,
        # never on running out of seats.
        while ended is None and step.seats:
            turn = step.seats[0].turn
            recorded = orders.get(turn)
            if recorded is None:
                stats["missing_turn"] += 1
                return
            acts = []
            for s in step.seats:
                a = recorded[s.seat]
                n = len(s.obs["mine"])
                if len(a) != n:        # a struck call: the seat's ants held
                    a = "-" * n
                    if s.seat in wanted:
                        stats["struck_turns"] += 1
                elif s.seat in wanted and n:
                    rows.append(json.dumps({
                        "o": s.obs, "a": a, "m": state.get(s.seat) or {},
                        "k": [rp["match_id"], s.seat, turn], "who": seat_model[s.seat],
                    }, separators=(",", ":")))
                state[s.seat] = remember(state.get(s.seat), s.obs)
                acts.append(a)
            step = env.step(acts)
            if step.ended:
                ended = step.ended[0]
        if ended is None or ended.scores != rp["scores"] or ended.turns != rp["turns"]:
            stats["mismatch"] += 1
            return
    for r in rows:
        out.write(r + "\n")
    stats["matches"] += 1
    stats["rows"] += len(rows)
    for seat in wanted:
        stats["seats_" + seat_model[seat]] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--models", nargs="+", default=["TigerAnts"], help="model names, or ALL for every seat")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    stats: Counter = Counter()
    paths = sorted(a.dir.glob("*.replay.json"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt") as out:
        first = json.loads(paths[0].read_text())
        out.write(json.dumps({
            "kind": "ladder-replays", "engine_digest": first["engine_digest"],
            "models": a.models, "teacher": "the ladder leaders' own recorded orders",
            # The leaders' moves were chosen by models that may carry memory of their own; the rows
            # carry planes.MEMORY so a memory model can learn to use what the seat had seen.
            "teacher_remembers": True, "memory": [m.name for m in MEMORY],
        }) + "\n")
        for i, p in enumerate(paths, 1):
            match = json.loads(p.with_name(p.name.replace(".replay", ".match")).read_text())
            replay(p, match, set(a.models), out, stats)
            if i % 20 == 0:
                print(f"{i}/{len(paths)} {dict(stats)}", flush=True)
    print(dict(stats))


if __name__ == "__main__":
    main()

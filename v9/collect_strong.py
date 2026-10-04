"""Behaviour-cloning data from the strong teacher, in collect.py's row format.

Two sources:

    # the strong teacher playing itself on the season's boards
    python collect_strong.py rollout --seat-turns 80000 --seed 11 --out data/strong-11.jsonl.gz
    # real ladder positions (every seat of every QA replay), relabelled with its moves
    python collect_strong.py relabel data/qa_all.jsonl.gz --out data/strong-relabel.jsonl.gz

Rows carry the seat's remembered state under `m`, which is also what the teacher reads, so
`train.bc --memory` learns labels that depend on the memory the model will carry.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

from tb_baselines.env import Env, orders_from_indices
from tb_baselines.planes import MEMORY, remember

sys.path.insert(0, str(Path(__file__).parent))
from strong_teacher import StrongTeacher  # noqa: E402

HERE = Path(__file__).parent


def header(engine: str, source: str) -> str:
    return json.dumps({
        "kind": "strong-teacher", "engine_digest": engine, "source": source,
        "teacher": "strong_teacher.StrongTeacher (defaults), remembering",
        "teacher_remembers": True, "memory": [m.name for m in MEMORY],
    }) + "\n"


def rollout(a) -> None:
    teacher = StrongTeacher()
    maps = str(HERE / "maps_hive")
    written, t0 = 0, time.time()
    state: dict = {}
    with Env(waves=8, matches_per_wave=2, max_turns=a.max_turns, seed=a.seed, maps=maps) as env, \
            gzip.open(a.out, "wt") as f:
        f.write(header(env.engine_digest, f"self-play rollout, seed {a.seed}"))
        step = env.reset()
        while written < a.seat_turns and step.seats:
            acts = []
            for s in step.seats:
                key = (s.ep, s.seat)
                st = state.get(key)
                if s.obs["mine"]:
                    moves = teacher.orders(s.obs, st)
                    order = orders_from_indices(np.asarray(moves), [len(s.obs["mine"])])[0]
                    f.write(json.dumps({"o": s.obs, "a": order, "m": st or {},
                                        "k": [s.ep, s.seat, s.turn]}, separators=(",", ":")) + "\n")
                    written += 1
                else:
                    order = ""
                acts.append(order)
                state[key] = remember(st, s.obs)
            step = env.step(acts)
            for e in step.ended:
                for k in range(e.seat_count):
                    state.pop((e.ep, k), None)
            if written and written % 10000 < len(acts):
                print(f"{written:,} rows  {written / (time.time() - t0):.0f}/s", flush=True)
    print(f"done: {written:,} rows in {time.time() - t0:.0f}s")


def dagger(a) -> None:
    """The student plays; the strong teacher labels every position the student reaches.

    `--opponents` fills the other seats (checkpoints or `strong`), so the student also meets
    positions that arise against stronger or different play; only the student's seats are written.
    """
    import torch
    from duel import make
    torch.set_num_threads(2)
    teacher = StrongTeacher()
    # --safe / --hill: the student plays as safe_policy.py exports it
    student = make(f"{'hill' if a.hill else 'safe' if a.safe else 'ckpt'}:{a.student}")
    table = [student] + [make(o) for o in a.opponents]
    maps = str(HERE / "maps_hive")
    written, t0 = 0, time.time()
    state: dict = {}
    with Env(waves=8, matches_per_wave=2, max_turns=a.max_turns, seed=a.seed, maps=maps) as env, \
            gzip.open(a.out, "wt") as f:
        f.write(header(env.engine_digest, f"DAgger: states of {a.student}, seed {a.seed}, "
                                          f"opponents {a.opponents or 'itself'}"))
        step = env.reset()
        while written < a.seat_turns and step.seats:
            acts = []
            for s in step.seats:
                key = (s.ep, s.seat)
                st = state.get(key)
                pol = table[(s.seat + s.ep) % len(table)]
                if not s.obs["mine"]:
                    acts.append("")
                else:
                    acts.append(pol.act(key, s.obs))
                    if pol is student:
                        moves = teacher.orders(s.obs, st)
                        label = orders_from_indices(np.asarray(moves), [len(s.obs["mine"])])[0]
                        f.write(json.dumps({"o": s.obs, "a": label, "m": st or {},
                                            "k": [s.ep, s.seat, s.turn]}, separators=(",", ":")) + "\n")
                        written += 1
                        if written % 10000 == 0:
                            print(f"{written:,} rows  {written / (time.time() - t0):.0f}/s", flush=True)
                state[key] = remember(st, s.obs)
            step = env.step(acts)
            for e in step.ended:
                for k in range(e.seat_count):
                    state.pop((e.ep, k), None)
                    for pol in table:
                        getattr(pol, "mem", {}).pop((e.ep, k), None)
                        getattr(pol, "state", {}).pop((e.ep, k), None)
    print(f"done: {written:,} rows in {time.time() - t0:.0f}s")


def relabel(a) -> None:
    teacher = StrongTeacher()
    written, t0 = 0, time.time()
    with gzip.open(a.data, "rt") as src, gzip.open(a.out, "wt") as f:
        h = json.loads(src.readline())
        f.write(header(h["engine_digest"], f"relabelled ladder positions from {a.data}"))
        for line in src:
            r = json.loads(line)
            if not r["o"]["mine"]:
                continue
            moves = teacher.orders(r["o"], r.get("m") or None)
            r["a"] = orders_from_indices(np.asarray(moves), [len(r["o"]["mine"])])[0]
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
            written += 1
            if written % 20000 == 0:
                print(f"{written:,} rows  {written / (time.time() - t0):.0f}/s", flush=True)
    print(f"done: {written:,} rows in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rollout")
    r.add_argument("--seat-turns", type=int, default=80000)
    r.add_argument("--max-turns", type=int, default=500)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--out", type=Path, required=True)
    l = sub.add_parser("relabel")
    l.add_argument("data", type=Path)
    l.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("dagger")
    d.add_argument("--student", type=Path, required=True)
    d.add_argument("--opponents", nargs="*", default=[])
    d.add_argument("--safe", action="store_true", help="the student plays with own-ant collisions removed")
    d.add_argument("--hill", action="store_true", help="--safe, and it takes an undefended enemy hill beside it")
    d.add_argument("--seat-turns", type=int, default=75000)
    d.add_argument("--max-turns", type=int, default=500)
    d.add_argument("--seed", type=int, default=21)
    d.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"rollout": rollout, "relabel": relabel, "dagger": dagger}[a.cmd](a)


if __name__ == "__main__":
    main()

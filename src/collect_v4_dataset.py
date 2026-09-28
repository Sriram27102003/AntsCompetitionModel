import argparse
import gzip
import json
import time
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tb_baselines.env import Env, orders_from_indices
from tb_baselines.teacher import known
from src.tactical_teacher_v4 import TacticalTeacherV4

def run_collection(out_path: str, seat_turns: int = 100000, waves: int = 4, matches_per_wave: int = 8, max_turns: int = 280, seed: int = 1337, maps: str = "season_maps"):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    teacher = TacticalTeacherV4()

    written = 0
    ants = 0
    episodes = 0
    started = time.time()
    last_report = started

    print(f"[*] Starting Tactical Teacher v4 (Hill Raider) data collection on '{maps}': targeting {seat_turns:,} seat-turns...", flush=True)

    with Env(waves=waves, matches_per_wave=matches_per_wave, max_turns=max_turns, seed=seed, maps=maps) as env, gzip.open(out, "wt") as f:
        header = {
            "kind": "tactical-rollout",
            "engine_digest": env.engine_digest,
            "evaluator": env.evaluator,
            "max_turns": max_turns,
            "env_seed": seed,
            "maps": maps,
            "teacher": "TacticalTeacherV4-HillRaider"
        }
        f.write(json.dumps(header) + "\n")

        step = env.reset()
        while written < seat_turns:
            actions = []
            for s in step.seats:
                water, seen = known(s.obs)
                moves = teacher.orders(s.obs, water, seen)
                order = orders_from_indices(moves, [len(s.obs["mine"])])[0]
                actions.append(order)
                if written < seat_turns and s.obs["mine"]:
                    f.write(json.dumps({"o": s.obs, "a": order}, separators=(",", ":")) + "\n")
                    written += 1
                    ants += len(order)

            step = env.step(actions)
            for e in step.ended:
                episodes += 1

            now = time.time()
            if now - last_report >= 8.0 or written >= seat_turns:
                pct = written / seat_turns * 100
                speed = written / (now - started)
                eta = (seat_turns - written) / speed if speed > 0 else 0
                print(f"    Progress: {written:,} / {seat_turns:,} seat-turns ({pct:.1f}%) | {speed:.1f} turns/s | ETA: {eta:.0f}s", flush=True)
                last_report = now

            if not step.seats:
                break

    elapsed = time.time() - started
    print(f"[+] Collection complete in {elapsed:.1f}s: {written:,} seat-turns, {ants:,} ant decisions across {episodes} episodes -> {out}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seat-turns", type=int, default=100000)
    parser.add_argument("--out", type=str, default="data/v4_hill_raider_100k.jsonl.gz")
    parser.add_argument("--maps", type=str, default="season_maps")
    parser.add_argument("--waves", type=int, default=4)
    parser.add_argument("--matches-per-wave", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=280)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    run_collection(args.out, seat_turns=args.seat_turns, maps=args.maps, waves=args.waves,
                   matches_per_wave=args.matches_per_wave, max_turns=args.max_turns, seed=args.seed)

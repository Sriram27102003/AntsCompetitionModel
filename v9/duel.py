"""Head-to-head harness: scripted teachers and trained checkpoints, in the Hive Minds engine.

    python duel.py strong ckpt:../runs/v5_mini/best.pt --maps 'tiny-*-2p-*' 'small*-2p-*' --games 24

A policy is `strong` (strong_teacher.py), `teacher` (the upstream remembering teacher), or
`ckpt:<path>` (a MiniTrunk(64,6) .pt, with or without memory). Seats rotate every episode, so each
policy plays every seat. Reports what the rating cares about: finishing order against each other
policy, plus colony size at turns 25 and 50 (FatAss's colony stalls at ~8 ants once contact starts).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tb_baselines import nets
from tb_baselines.env import Env, orders_from_indices
from tb_baselines.export import classes
from tb_baselines.planes import MEMORY_SOURCES, encode, encode_memory, remember
from tb_baselines.teacher import Teacher, known
from tb_baselines.train.bc import override

sys.path.insert(0, str(Path(__file__).parent))
from strong_teacher import StrongTeacher  # noqa: E402

HERE = Path(__file__).parent


def hill_table() -> dict:
    """(rows, cols, an own hill) -> every enemy hill on that board, for the season's boards.

    A board is one design repeated for every seat by a fixed shift, so a seat that sees one of its
    own hills on a known board knows where every enemy hill is from turn 0. Entries two boards
    disagree on are dropped.
    """
    import json
    table: dict = {}
    for f in sorted((HERE / "maps_hive").glob("*.json")):
        m = json.loads(f.read_text())
        rows, cols, n = m["rows"], m["cols"], m["players"]
        dr, dc = m["symmetry"]["dr"], m["symmetry"]["dc"]
        hills = {tuple(h) for h in m["hills"]}
        seen: set = set()
        owner: dict = {}
        for h in m["hills"]:
            h = tuple(h)
            if h in seen:
                continue
            for k in range(n):
                q = ((h[0] + k * dr) % rows, (h[1] + k * dc) % cols)
                owner[q] = k
                seen.add(q)
        for h, k in owner.items():
            foes = sorted(q for q in hills if owner[q] != k)
            key = (rows, cols, h)
            table[key] = None if key in table and table[key] != foes else foes
    return {k: v for k, v in table.items() if v}


class Scripted:
    def __init__(self, kind: str):
        self.name = kind
        # `strong:loss_behind=1.0,raze_min_ants=12` passes settings to the teacher
        base, _, args = kind.partition(":")
        num = lambda v: float(v) if "." in v else int(v) if v.lstrip("-").isdigit() else v
        kw = {k: num(v) for k, v in (x.split("=") for x in args.split(",") if x)}
        # `know_hills=1`: the seat remembers every enemy hill of a known board from the first turn
        self.table = hill_table() if kw.pop("know_hills", 0) else None
        self.kind = base
        self.bot = StrongTeacher(**kw) if base == "strong" else Teacher()
        self.state: dict = {}

    def act(self, key, obs) -> str:
        st = self.state.get(key)
        if self.table is not None and st is None:
            rows, cols = obs["size"]
            for r, c, o in obs["hills"]:
                foes = self.table.get((rows, cols, (r, c))) if o == 0 else None
                if foes:
                    st = {"food_seen": [], "hill_foe_seen": [list(h) for h in foes]}
                    break
        if self.kind == "strong":
            moves = self.bot.orders(obs, st)
        else:
            water, seen = known(obs)
            moves = self.bot.orders(obs, water, seen, st)
        self.state[key] = remember(st, obs)
        return orders_from_indices(np.asarray(moves), [len(obs["mine"])])[0]


class Checkpoint:
    def __init__(self, path: str):
        self.name = Path(path).parent.name
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.memory = bool(ck.get("memory"))

        class A:       # the width is read off the weights, so a widened net loads like any other
            arch, blocks = "mini_trunk", 6
            channels = next(v for k, v in ck["trunk"].items() if k.endswith("mix.weight")).shape[0]
        self.net = nets.build(override(classes()["mini"], A), memory=self.memory).eval()
        self.net.load_state_dict(ck["trunk"])
        self.mem: dict = {}

    @torch.no_grad()
    def act(self, key, obs) -> str:
        board = torch.from_numpy(encode(obs))
        if self.memory:
            m = self.mem.get(key)
            if m is None:
                m = torch.from_numpy(encode_memory(None, obs["size"]))
            policy, m2 = self.net(board, m)
            self.mem[key] = m2
        else:
            policy = self.net(board)
        mine = np.array(obs["mine"], dtype=np.int64).reshape(-1, 2)
        logits = policy[0, :, mine[:, 0], mine[:, 1]]          # [5, ants]
        return orders_from_indices(logits.argmax(0).numpy(), [len(mine)])[0]


DELTAS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)])


def no_collisions(mine: np.ndarray, logits: np.ndarray, obs: dict, take_hills: bool = False,
                  reroute: bool = False) -> np.ndarray:
    """The policy's moves with own-ant collisions taken out: two ants ending on one square both die.

    A move into water or food is a hold (the engine ignores it). Where several ants would end on one
    square, an ant already holding there keeps it, else the most confident mover does, and the rest
    hold; holding can block someone else, so it repeats until nothing changes.
    """
    rows, cols = obs["size"]
    water, _ = known(obs)
    blocked = np.asarray(water, dtype=bool).copy()
    for r, c in obs["food"]:
        blocked[r, c] = True
    moves = logits.argmax(0)
    conf = logits.max(0)
    if take_hills:
        # An ant beside an enemy hill no enemy can defend this turn (none within d2 <= 10 of it:
        # attack range after one step) takes it: +2, and the only thing rank counts.
        foes = np.array([f[:2] for f in obs["foes"]], dtype=np.int64).reshape(-1, 2)
        for hr, hc, owner in obs["hills"]:
            if owner == 0:
                continue
            if len(foes):
                dr = np.abs(foes[:, 0] - hr); dr = np.minimum(dr, rows - dr)
                dc = np.abs(foes[:, 1] - hc); dc = np.minimum(dc, cols - dc)
                if (dr * dr + dc * dc <= 10).any():
                    continue
            for d in range(4):
                i = np.flatnonzero((((mine + DELTAS[d]) % (rows, cols)) == (hr, hc)).all(1))
                if len(i):
                    moves[i[0]] = d
                    conf[i[0]] = 1e4 - d
    if reroute:
        # Before anyone is made to hold, an ant that is blocked or loses a square takes its next
        # best move instead: the lost move is struck from its scores and it chooses again.
        left = logits.copy()
        left[moves, np.arange(len(moves))] = conf                 # a forced hill keeps its priority
        for _ in range(8):
            moves = left.argmax(0)
            conf = left.max(0)
            dest = (mine + DELTAS[moves]) % (rows, cols)
            stuck = blocked[dest[:, 0], dest[:, 1]] & (moves != 4)
            dest[stuck] = mine[stuck]
            key = dest[:, 0] * cols + dest[:, 1]
            uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            lose = stuck.copy()
            for g in np.flatnonzero(cnt > 1):
                ants = np.flatnonzero(inv == g)
                holder = ants[(moves[ants] == 4) | stuck[ants]]
                keep = holder[0] if len(holder) else ants[conf[ants].argmax()]
                for i in ants:
                    if i != keep and moves[i] != 4:
                        lose[i] = True
            if not lose.any():
                break
            left[moves[lose], np.flatnonzero(lose)] = -1e4
        moves = left.argmax(0)
        conf = left.max(0)
    for _ in range(16):
        dest = (mine + DELTAS[moves]) % (rows, cols)
        stuck = blocked[dest[:, 0], dest[:, 1]] & (moves != 4)
        moves[stuck] = 4
        dest[stuck] = mine[stuck]
        key = dest[:, 0] * cols + dest[:, 1]
        uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        if (cnt <= 1).all():
            break
        for g in np.flatnonzero(cnt > 1):
            ants = np.flatnonzero(inv == g)
            holder = ants[moves[ants] == 4]
            keep = holder[0] if len(holder) else ants[conf[ants].argmax()]
            for i in ants:
                if i != keep:
                    moves[i] = 4
    return moves


class SafeCheckpoint(Checkpoint):
    """`safe:<path>`: a checkpoint whose orders pass through `no_collisions`."""

    take_hills = False
    reroute = False

    def __init__(self, path: str):
        super().__init__(path)
        self.name += "+safe"

    @torch.no_grad()
    def act(self, key, obs) -> str:
        board = torch.from_numpy(encode(obs))
        m = self.mem.get(key)
        if m is None:
            m = torch.from_numpy(encode_memory(None, obs["size"]))
        policy, self.mem[key] = self.net(board, m)
        mine = np.array(obs["mine"], dtype=np.int64).reshape(-1, 2)
        logits = policy[0, :, mine[:, 0], mine[:, 1]].numpy()
        return orders_from_indices(no_collisions(mine, logits, obs, self.take_hills, self.reroute),
                                   [len(mine)])[0]


class HillCheckpoint(SafeCheckpoint):
    """`hill:<path>`: `safe:` plus taking an undefended enemy hill an ant stands beside."""

    take_hills = True

    def __init__(self, path: str):
        super().__init__(path)
        self.name += "+hill"


class RouteCheckpoint(HillCheckpoint):
    """`route:<path>`: `hill:` plus a blocked ant taking its next best move before it is held."""

    reroute = True

    def __init__(self, path: str):
        super().__init__(path)
        self.name += "+route"


def make(spec: str):
    if spec.startswith("route:"):
        return RouteCheckpoint(spec[6:])
    if spec.startswith("hill:"):
        return HillCheckpoint(spec[5:])
    if spec.startswith("safe:"):
        return SafeCheckpoint(spec[5:])
    return Checkpoint(spec[5:]) if spec.startswith("ckpt:") else Scripted(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="+")
    ap.add_argument("--maps", nargs="+", default=["*"], help="globs over maps_hive/")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--seed", type=int, default=500)
    a = ap.parse_args()
    torch.set_num_threads(4)

    pols = [make(p) for p in a.policies]
    names = [p.name for p in pols]
    maps = sorted({str(f) for g in a.maps for f in (HERE / "maps_hive").glob(g + ".json")})
    if not maps:
        raise SystemExit("no maps match")
    import itertools
    two_seat = all("-2p-" in Path(m).name for m in maps)
    pairs = list(itertools.permutations(range(len(pols)), 2))

    def who(ep, seat):
        # On two-seat boards every ordered pair plays in turn, so each policy meets every other in
        # both seats; on larger boards the policies rotate around the table.
        if two_seat:
            return pairs[ep % len(pairs)][seat]
        return (seat + ep) % len(pols)

    h2h = defaultdict(lambda: [0, 0, 0])                       # (a, b) -> ahead, tied, behind
    first = defaultdict(int)
    games = defaultdict(int)
    size = defaultdict(lambda: defaultdict(list))              # name -> turn -> ants
    # Rank is score only: +2 per enemy hill razed, -1 per own hill lost. Tracked per seat from the
    # running scores, so a result can be read as points won and points given away.
    last: dict = {}                                            # (ep, seat) -> score last turn
    tally: dict = {}                                           # (ep, seat) -> [razed, lost, first raze turn]
    points = defaultdict(lambda: {"score": [], "razed": [], "lost": [], "first": [], "turns": []})
    reasons = defaultdict(int)
    t0 = time.time()
    done = 0
    # A wave plays one board, so one wave per board (up to 16) spreads the games across them.
    waves = min(len(maps), 16)
    with Env(waves=waves, matches_per_wave=max(1, a.parallel // waves), max_turns=a.max_turns, seed=a.seed,
             maps=maps) as env:
        step = env.reset()
        while done < a.games and step.seats:
            acts = []
            for s in step.seats:
                p = pols[who(s.ep, s.seat)]
                if s.turn in (25, 50):
                    size[p.name][s.turn].append(len(s.obs["mine"]))
                acts.append(p.act((s.ep, s.seat), s.obs) if s.obs["mine"] else "")
            turn_now = step.seats[0].turn
            step = env.step(acts)

            def count(key, score):
                d = score - last.get(key, score)
                t = tally.setdefault(key, [0, 0, None])
                if d > 0:                                      # 2 x razed - lost, smallest reading
                    t[0] += (d + 1) // 2
                    t[1] += d % 2
                    if t[2] is None:
                        t[2] = turn_now
                elif d < 0:
                    t[1] += -d
                last[key] = score

            for ep, sc in step.scores.items():
                for k, v in enumerate(sc):
                    count((ep, k), v)
            for e in step.ended:
                done += 1
                reasons[e.reason] += 1
                for k in range(e.seat_count):
                    count((e.ep, k), e.scores[k])
                    razed, lost, when = tally.pop((e.ep, k), [0, 0, None])
                    last.pop((e.ep, k), None)
                    pt = points[pols[who(e.ep, k)].name]
                    pt["score"].append(e.scores[k]); pt["razed"].append(razed); pt["lost"].append(lost)
                    pt["turns"].append(e.turns)
                    if when is not None:
                        pt["first"].append(when)
                seats = [(pols[who(e.ep, k)].name, e.ranks[k]) for k in range(e.seat_count)]
                for na, ra in seats:
                    games[na] += 1
                    first[na] += ra == 1 and sum(r == 1 for _, r in seats) == 1
                    for nb, rb in seats:
                        if na != nb:
                            h = h2h[(na, nb)]
                            h[0 if ra < rb else 1 if ra == rb else 2] += 1
                print(f"[{done}/{a.games}] {e.map_id:26} {e.reason:15} {e.turns:4}t  "
                      + "  ".join(f"{n}={r}" for n, r in seats), flush=True)

    print(f"\n{done} games in {time.time() - t0:.0f}s")
    for n in dict.fromkeys(names):
        s25, s50 = size[n][25], size[n][50]
        print(f"{n:18} seats {games[n]:4}  sole first {first[n] / max(games[n], 1):4.0%}  "
              f"ants@25 {np.mean(s25) if s25 else 0:5.1f}  ants@50 {np.mean(s50) if s50 else 0:5.1f}")
    print("points (per game: +2 a hill razed, -1 a hill lost):")
    for n in dict.fromkeys(names):
        pt = points[n]
        if pt["score"]:
            print(f"  {n:18} score {np.mean(pt['score']):5.2f}  razed {np.mean(pt['razed']):4.2f}  "
                  f"lost {np.mean(pt['lost']):4.2f}  razed nothing {np.mean(np.array(pt['razed']) == 0):4.0%}  "
                  f"first raze turn {np.median(pt['first']) if pt['first'] else float('nan'):5.0f}  "
                  f"game length {np.median(pt['turns']):4.0f}")
    print("endings: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))
    print("head to head (ahead / tied / behind):")
    for (na, nb), (w, t, l) in sorted(h2h.items()):
        print(f"  {na:18} vs {nb:18} {w:3} / {t:3} / {l:3}")


if __name__ == "__main__":
    main()

"""A stronger scripted teacher for Hive Minds: economy first, and fights only where the focus rule
says we come out ahead.

The upstream teacher (tb_baselines/teacher.py) is one BFS potential field and has no idea what a
fight is, which is exactly how FatAss loses to FireAnts: once contact starts (~turn 20) its colony
stops growing. This one plans each turn in layers, each claiming ants the next cannot use:

1. Hill guard   - enough ants near each own hill to answer the foes near it, standing beside the
                  hill (never on it: an ant on its own hill blocks the spawn).
2. Food         - one ant per food item, nearest first, over the known map (food refuses a move, so
                  the target is a square next to it).
3. Raze         - enemy hills seen now or remembered, when we have the ants to spare.
4. Explore      - the rest spread toward what the seat has not seen.
5. Combat       - every ant that could be in attack range after this turn is re-planned together:
                  coordinate descent over their five moves, each joint move scored by the exact
                  focus rule against the worst of three enemy responses (hold, advance, retreat).
6. Resolve      - destinations are reserved so no two of our ants ever share a square.

It is deterministic (fixed tie order), so its labels are a function of the observation and the
seat's memory, which is what makes it imitable.

    from strong_teacher import StrongTeacher
    moves = StrongTeacher().orders(obs, state)   # state: planes.remember() output or None
"""
from __future__ import annotations

import numpy as np

from tb_baselines.teacher import known

DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1), (0, 0))   # N E S W -, the channel order
HOLD = 4
ATTACK2 = 5
FAR = 1 << 20


def wrap_d2(a: np.ndarray, b: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Pairwise wrapped squared distance between [n,2] and [m,2] integer positions."""
    dr = np.abs(a[:, None, 0] - b[None, :, 0])
    dc = np.abs(a[:, None, 1] - b[None, :, 1])
    dr = np.minimum(dr, rows - dr)
    dc = np.minimum(dc, cols - dc)
    return dr * dr + dc * dc


def bfs(sources: list[tuple[int, int]], passable: np.ndarray, depth: int) -> np.ndarray:
    """Distance field from every source at once, over passable cells, wrapping, up to `depth`."""
    rows, cols = passable.shape
    dist = np.full((rows, cols), FAR, dtype=np.int32)
    front = np.zeros((rows, cols), dtype=bool)
    for r, c in sources:
        front[r, c] = True
    dist[front] = 0
    seen = front.copy()
    for d in range(1, depth + 1):
        grow = (np.roll(front, 1, 0) | np.roll(front, -1, 0) |
                np.roll(front, 1, 1) | np.roll(front, -1, 1))
        front = grow & passable & ~seen
        if not front.any():
            break
        dist[front] = d
        seen |= front
    return dist


def step_down(dist: np.ndarray, r: int, c: int, blocked: np.ndarray) -> int:
    """The move (0..3) that lowers `dist` most from (r, c), or HOLD when none does."""
    rows, cols = dist.shape
    best, move = dist[r, c], HOLD
    for k, (dr, dc) in enumerate(DELTAS[:4]):
        rr, cc = (r + dr) % rows, (c + dc) % cols
        if not blocked[rr, cc] and dist[rr, cc] < best:
            best, move = dist[rr, cc], k
    return move


class StrongTeacher:
    def __init__(self, food_depth: int = 40, guard_radius2: int = 30, raze_min_ants: int = 8,
                 fight_radius2: int = 20, passes: int = 2, loss_ahead: float = 1.0,
                 loss_behind: float = 1.6, spare: str = "field", take_hill: int = 0):
        self.food_depth = food_depth
        self.guard_radius2 = guard_radius2
        self.raze_min_ants = raze_min_ants
        self.fight_radius2 = fight_radius2
        self.passes = passes
        self.loss_ahead = loss_ahead
        self.loss_behind = loss_behind
        # "own": spare ants raze, then explore the frontier. "field": spare ants follow the upstream
        # teacher's potential field (food, remembered hills, frontier in one flood), which grows a
        # colony faster in the opening.
        self.spare = spare
        # 1: an ant next to an enemy hill steps onto it whenever it would survive there with the
        # enemy holding still. A razed hill is +2 and the only thing rank counts; without this the
        # fight step can talk the ant out of the last square. 0 keeps the labels of earlier datasets.
        self.take_hill = take_hill
        self._field = None

    # ---- the turn -----------------------------------------------------------------------------

    def orders(self, obs: dict, state: dict | None = None) -> np.ndarray:
        rows, cols = obs["size"]
        mine = np.array(obs["mine"], dtype=np.int64).reshape(-1, 2)
        n = len(mine)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        foes = np.array([f[:2] for f in obs["foes"]], dtype=np.int64).reshape(-1, 2)
        water, seen = known(obs)
        water = water.astype(bool)
        seen = seen.astype(bool)
        food = np.zeros((rows, cols), dtype=bool)
        for r, c in obs["food"]:
            food[r, c] = True
        passable = ~water & ~food                       # food refuses a move
        own_hills = [(r, c) for r, c, o in obs["hills"] if o == 0]
        foe_hills = {(r, c) for r, c, o in obs["hills"] if o != 0}
        vis = seen                                      # `known` folds this turn's vis into seen
        for r, c in (state or {}).get("hill_foe_seen", []):
            if not (vis[r, c] and (r, c) not in foe_hills):
                foe_hills.add((r, c))                   # remembered, and not seen to be gone

        target = [None] * n                             # a distance field per ant
        free = np.ones(n, dtype=bool)
        cell = {tuple(p): i for i, p in enumerate(mine.tolist())}

        # 1. Hill guard: as many guards as foes near the hill, drawn from the nearest free ants.
        if len(foes) and own_hills:
            hills = np.array(own_hills, dtype=np.int64)
            near = wrap_d2(hills, foes, rows, cols) <= self.guard_radius2
            for h, (hr, hc) in enumerate(own_hills):
                need = int(near[h].sum())
                if not need:
                    continue
                ring = [((hr + dr) % rows, (hc + dc) % cols) for dr, dc in DELTAS[:4]]
                ring = [p for p in ring if passable[p]]
                if not ring:
                    continue
                field = bfs(ring, passable, 20)
                order = sorted((field[r, c], i) for i, (r, c) in enumerate(mine.tolist())
                               if free[i] and field[r, c] < FAR)
                for _, i in order[:need]:
                    target[i] = field
                    free[i] = False

        # 2. Food: one ant per item, nearest pairs first.
        pairs = []
        fields = {}
        for fr, fc in obs["food"]:
            nb = [((fr + dr) % rows, (fc + dc) % cols) for dr, dc in DELTAS[:4]]
            nb = [p for p in nb if passable[p]]
            if not nb:
                continue
            f = bfs(nb, passable, self.food_depth)
            fields[(fr, fc)] = f
            for i, (r, c) in enumerate(mine.tolist()):
                if free[i] and f[r, c] < FAR:
                    pairs.append((int(f[r, c]), i, (fr, fc)))
        pairs.sort()
        taken = set()
        for d, i, fd in pairs:
            if free[i] and fd not in taken:
                taken.add(fd)
                free[i] = False
                target[i] = fields[fd]
        # Remembered food spots out of sight are weak targets for idle ants.
        spots = [(r, c) for r, c in (state or {}).get("food_seen", [])
                 if not vis[r, c] and passable[r, c]]

        field_moves = None
        if self.spare == "field" and free.any():
            if self._field is None:
                from tb_baselines.teacher import Teacher
                self._field = Teacher()
            field_moves = np.asarray(self._field.orders(obs, water,
                                                        seen, state))
            free_field = free.copy()
            free[:] = False

        # 3. Raze: send spare ants at known enemy hills once the colony can afford it.
        if foe_hills and free.sum() and n >= self.raze_min_ants:
            f = bfs(sorted(foe_hills), ~water, 120)
            for i in np.flatnonzero(free):
                r, c = mine[i]
                if f[r, c] < FAR:
                    target[i] = f
                    free[i] = False

        # 4. Explore: the frontier of what this seat has never seen, then remembered food spots.
        if free.any():
            frontier = ~seen & ~water
            srcs = [tuple(p) for p in np.argwhere(frontier)] or spots
            if srcs:
                f = bfs(srcs, passable | frontier, 200)
                for i in np.flatnonzero(free):
                    target[i] = f

        plan = np.full(n, HOLD, dtype=np.int64)
        for i, (r, c) in enumerate(mine.tolist()):
            if target[i] is not None:
                plan[i] = step_down(target[i], r, c, ~passable)
        if field_moves is not None:
            plan[free_field] = field_moves[free_field]
        # Never idle on our own hill: the spawn waits for the square.
        for i, (r, c) in enumerate(mine.tolist()):
            if plan[i] == HOLD and (r, c) in set(own_hills):
                for k, (dr, dc) in enumerate(DELTAS[:4]):
                    if passable[(r + dr) % rows, (c + dc) % cols]:
                        plan[i] = k
                        break

        # 5. Combat: re-plan every ant that could be within attack range after this turn.
        if len(foes):
            plan = self.fight(mine, foes, plan, passable, rows, cols, n, len(foes))

        if self.take_hill and foe_hills:
            plan = self.raze_now(mine, foes, plan, passable, foe_hills, rows, cols)

        # 6. Resolve: no two of our ants end on one square.
        return self.resolve(mine, plan, passable, rows, cols)

    # ---- combat -------------------------------------------------------------------------------

    def fight(self, mine, foes, plan, passable, rows, cols, n_mine, n_foes):
        d2 = wrap_d2(mine, foes, rows, cols)
        involved = np.flatnonzero(d2.min(axis=1) <= self.fight_radius2)
        if not len(involved):
            return plan
        foes_near = foes[d2[involved].min(axis=0) <= self.fight_radius2]
        # Trade freely when the colony is ahead of what it can see; carefully when it is not.
        loss_w = self.loss_ahead if n_mine >= 1.5 * n_foes else self.loss_behind
        scen = self.enemy_moves(foes_near, mine, passable, rows, cols)

        def dest(i, k):
            r, c = mine[i]
            dr, dc = DELTAS[k]
            rr, cc = (r + dr) % rows, (c + dc) % cols
            return (rr, cc) if passable[rr, cc] else (r, c)

        cur = plan.copy()

        def score(moves) -> float:
            pos = np.array([dest(i, moves[i]) for i in involved], dtype=np.int64)
            worst = None
            for e in scen:
                kills, losses = battle(pos, e, rows, cols)
                s = kills - loss_w * losses
                worst = s if worst is None else min(worst, s)
            return worst

        base = score(cur)
        for _ in range(self.passes):
            for i in involved:
                best_k, best_s = cur[i], base
                for k in range(5):
                    if k == cur[i]:
                        continue
                    trial = cur.copy()
                    trial[i] = k
                    # keep a tiny preference for the planned move, so ties keep gathering
                    s = score(trial) - (0.0 if k == plan[i] else 0.01)
                    if s > best_s + 1e-9:
                        best_k, best_s = k, s
                cur[i] = best_k
                base = best_s if best_k != plan[i] else score(cur)
        return cur

    @staticmethod
    def raze_now(mine, foes, plan, passable, foe_hills, rows, cols):
        """Step onto an adjacent enemy hill when the ant lives through this turn's fight there."""
        plan = plan.copy()
        size = np.array([rows, cols])
        foe_set = {tuple(f) for f in foes.tolist()}
        done = set()
        for i, (r, c) in enumerate(mine.tolist()):
            for k, (dr, dc) in enumerate(DELTAS[:4]):
                h = ((r + dr) % rows, (c + dc) % cols)
                if h not in foe_hills or h in foe_set or h in done or not passable[h]:
                    continue
                trial = plan.copy()
                trial[i] = k
                pos = (mine + np.array(DELTAS)[trial]) % size
                stay = ~passable[pos[:, 0], pos[:, 1]]
                pos[stay] = mine[stay]
                if len(foes):
                    inr = wrap_d2(pos, foes, rows, cols) <= ATTACK2
                    f_ours, f_theirs = inr.sum(axis=1), inr.sum(axis=0)
                    if (inr[i] & (f_theirs <= f_ours[i])).any():
                        continue                               # it would die on the hill: no raze
                plan = trial
                done.add(h)
                break
        return plan

    @staticmethod
    def enemy_moves(foes, mine, passable, rows, cols):
        """Three enemy responses: hold, everyone steps toward our nearest ant, everyone away."""
        out = [foes]
        d = wrap_d2(foes, mine, rows, cols)
        near = mine[d.argmin(axis=1)]
        for sign in (1, -1):
            moved = []
            for (r, c), (tr, tc) in zip(foes.tolist(), near.tolist()):
                best, bp = None, (r, c)
                for dr, dc in DELTAS[:4]:
                    rr, cc = (r + dr) % rows, (c + dc) % cols
                    if not passable[rr, cc]:
                        continue
                    dd = int(wrap_d2(np.array([[rr, cc]]), np.array([[tr, tc]]), rows, cols)[0, 0])
                    key = sign * dd
                    if best is None or key < best:
                        best, bp = key, (rr, cc)
                moved.append(bp)
            out.append(np.array(moved, dtype=np.int64).reshape(-1, 2))
        return out

    # ---- collisions ---------------------------------------------------------------------------

    @staticmethod
    def resolve(mine, plan, passable, rows, cols):
        n = len(mine)
        moves = plan.copy()
        for _ in range(4):
            dest = []
            for i, (r, c) in enumerate(mine.tolist()):
                dr, dc = DELTAS[moves[i]]
                rr, cc = (r + dr) % rows, (c + dc) % cols
                dest.append((rr, cc) if passable[rr, cc] else (r, c))
            claims = {}
            for i, d in enumerate(dest):
                claims.setdefault(d, []).append(i)
            clash = [ids for ids in claims.values() if len(ids) > 1]
            if not clash:
                break
            for ids in clash:
                # the ant already standing there keeps the square; the others hold instead
                keep = next((i for i in ids if tuple(mine[i]) == dest[i]), ids[0])
                for i in ids:
                    if i != keep:
                        moves[i] = HOLD
        return moves


def battle(ours: np.ndarray, theirs: np.ndarray, rows: int, cols: int) -> tuple[int, int]:
    """(enemies killed, ours lost) by the focus rule: an ant dies if any enemy in range has focus
    less than or equal to its own. Every focus is counted before anyone is removed."""
    if not len(ours) or not len(theirs):
        return 0, 0
    inr = wrap_d2(ours, theirs, rows, cols) <= ATTACK2
    f_ours = inr.sum(axis=1)
    f_theirs = inr.sum(axis=0)
    ours_die = (inr & (f_theirs[None, :] <= f_ours[:, None])).any(axis=1)
    theirs_die = (inr & (f_ours[:, None] <= f_theirs[None, :])).any(axis=0)
    return int(theirs_die.sum()), int(ours_die.sum())

"""Grandmaster Tactical Teacher v6: Two-Phase Gated Architecture (v2 Economy + v5 Room Siege)."""
from __future__ import annotations
import numpy as np

N, E, S, W, HOLD = 0, 1, 2, 3, 4
DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
MOVE_CHARS = ("N", "E", "S", "W", "-")

FAR = 9999
ATTACK_RADIUS2 = 5

def wrap_dist2(r1: int, c1: int, r2: int, c2: int, rows: int, cols: int) -> int:
    dr = abs(r1 - r2)
    dr = min(dr, rows - dr)
    dc = abs(c1 - c2)
    dc = min(dc, cols - dc)
    return dr * dr + dc * dc

class TacticalTeacherV6:
    def __init__(self, max_depth: int = 72):
        self.max_depth = max_depth

    def compute_field(
        self,
        obs: dict,
        water: np.ndarray,
        seen: np.ndarray,
        target_hills: list[tuple[int, int]],
        target_food: list[tuple[int, int]],
        threat_mask: np.ndarray | None,
        is_siege_phase: bool
    ) -> np.ndarray:
        rows, cols = obs["size"]
        dist = np.full((rows, cols), FAR, dtype=np.int32)
        passable = ~water

        at_level: dict[int, np.ndarray] = {}

        def seed(mask: np.ndarray, level: int):
            if level in at_level:
                at_level[level] |= mask
            else:
                at_level[level] = mask.copy()

        # Phase-based prioritization:
        if is_siege_phase:
            # PHASE 2 (Army >= 25 ants): Enemy hills are top priority (Level 0)
            if target_hills:
                h_grid = np.zeros((rows, cols), dtype=bool)
                for r, c in target_hills:
                    h_grid[r, c] = True
                seed(h_grid & passable, 0)
            # Food is secondary (Level 12)
            if target_food:
                f_grid = np.zeros((rows, cols), dtype=bool)
                for r, c in target_food:
                    f_grid[r, c] = True
                seed(f_grid & passable, 12)
        else:
            # PHASE 1 (Economy < 25 ants, v2 style): Food is top priority (Level 0)
            if target_food:
                f_grid = np.zeros((rows, cols), dtype=bool)
                for r, c in target_food:
                    f_grid[r, c] = True
                seed(f_grid & passable, 0)
            # Enemy hills secondary exploration (Level 14)
            if target_hills:
                h_grid = np.zeros((rows, cols), dtype=bool)
                for r, c in target_hills:
                    h_grid[r, c] = True
                seed(h_grid & passable, 14)

        # Defensive threat interception (Level 2)
        if threat_mask is not None and threat_mask.any():
            seed(threat_mask & passable, 2)

        # Unseen fog frontier (Level 18)
        unseen = ~seen
        if unseen.any():
            seed(unseen & passable, 18)

        if not at_level:
            return dist

        wave = np.zeros((rows, cols), dtype=bool)
        for depth in range(0, self.max_depth + 1):
            step = at_level.pop(depth, None)
            if wave.any():
                spread = (
                    np.roll(wave, 1, 0) | np.roll(wave, -1, 0)
                    | np.roll(wave, 1, 1) | np.roll(wave, -1, 1)
                )
                step = spread if step is None else (spread | step)
            if step is None:
                continue
            step &= passable & (dist == FAR)
            if not step.any() and not at_level:
                break
            dist[step] = depth
            wave = step

        return dist

    def orders(self, obs: dict, water: np.ndarray, seen: np.ndarray) -> np.ndarray:
        mine = obs["mine"]
        if not mine:
            return np.zeros(0, dtype=np.int64)

        rows, cols = obs["size"]
        foes = obs.get("foes", [])
        hills = obs.get("hills", [])

        my_hills = [(h[0], h[1]) for h in hills if h[2] == 0]
        enemy_hills = [(h[0], h[1]) for h in hills if h[2] != 0]

        # Phase Gate: Siege mode only unlocks at 25+ ants with visible enemy hills
        is_siege_phase = (len(mine) >= 25) and (len(enemy_hills) > 0)

        threat_mask = np.zeros((rows, cols), dtype=bool)
        for er, ec, _ in foes:
            for hr, hc in my_hills:
                if wrap_dist2(er, ec, hr, hc, rows, cols) <= 144:
                    threat_mask[er, ec] = True
                    break

        food = [(f[0], f[1]) for f in obs.get("food", [])]
        dist = self.compute_field(obs, water, seen, enemy_hills, food, threat_mask, is_siege_phase)

        hill_dist = np.full((rows, cols), FAR, dtype=np.int32)
        if is_siege_phase:
            h_grid = np.zeros((rows, cols), dtype=bool)
            for hr, hc in enemy_hills:
                h_grid[hr, hc] = True
            wave = h_grid & (~water)
            for d in range(self.max_depth + 1):
                if d == 0:
                    step = wave
                else:
                    step = (np.roll(wave, 1, 0) | np.roll(wave, -1, 0) |
                            np.roll(wave, 1, 1) | np.roll(wave, -1, 1)) & (~water) & (hill_dist == FAR)
                if not step.any():
                    break
                hill_dist[step] = d
                wave = step

        blocked = np.zeros((rows, cols), dtype=bool)
        for r, c in food:
            blocked[r, c] = True

        out = np.full(len(mine), HOLD, dtype=np.int64)
        taken = {tuple(a) for a in mine}
        foe_positions = [(fr, fc) for fr, fc, _ in foes]

        def ant_priority(i: int):
            ar, ac = mine[i]
            on_hill = (ar, ac) in my_hills
            return (0 if on_hill else 1, dist[ar, ac], i)

        order = sorted(range(len(mine)), key=ant_priority)

        for i in order:
            ar, ac = mine[i]
            on_my_hill = (ar, ac) in my_hills

            valid_moves = []
            for k in range(4):
                dr, dc = DELTAS[k]
                nr, nc = (ar + dr) % rows, (ac + dc) % cols

                if (nr, nc) in my_hills and (nr, nc) not in foe_positions:
                    continue

                if water[nr, nc] or blocked[nr, nc] or (nr, nc) in taken:
                    continue
                valid_moves.append((k, nr, nc))

            can_hold = not on_my_hill and (ar, ac) not in taken
            if can_hold:
                valid_moves.append((HOLD, ar, ac))

            if not valid_moves:
                out[i] = HOLD
                continue

            best_move = HOLD
            best_score = -1e9

            near_enemy_hill = is_siege_phase and (hill_dist[ar, ac] <= 25)

            for k, dest_r, dest_c in valid_moves:
                enemy_threats = 0
                for fr, fc in foe_positions:
                    if wrap_dist2(dest_r, dest_c, fr, fc, rows, cols) <= ATTACK_RADIUS2:
                        enemy_threats += 1

                friendly_support = 0
                for oi, (mr, mc) in enumerate(mine):
                    if oi != i and wrap_dist2(dest_r, dest_c, mr, mc, rows, cols) <= ATTACK_RADIUS2:
                        friendly_support += 1

                is_razing_move = (dest_r, dest_c) in enemy_hills

                # COMBAT HEURISTICS (Fused v2 + v5)
                if is_razing_move:
                    combat_penalty = 25000.0
                elif enemy_threats == 0:
                    combat_penalty = 0.0
                elif friendly_support > enemy_threats:
                    combat_penalty = 500.0  # Decisive gank advantage
                elif friendly_support == enemy_threats and is_siege_phase and near_enemy_hill:
                    combat_penalty = 100.0  # Push through on siege
                else:
                    # STRICT ANT PRESERVATION (v2 superpower): Never throw an ant away
                    combat_penalty = -7000.0

                target_d = dist[dest_r, dest_c]
                strategic_score = -float(target_d)

                if near_enemy_hill:
                    h_d = hill_dist[dest_r, dest_c]
                    strategic_score += (25 - h_d) * 12.0

                # Strict home-hill clearance to ensure continuous spawning
                if on_my_hill and k != HOLD:
                    strategic_score += 1500.0

                tie_break = (4 - k) * 0.01
                total_score = combat_penalty + strategic_score + tie_break

                if total_score > best_score:
                    best_score = total_score
                    best_move = k

            out[i] = best_move
            if best_move != HOLD:
                dr, dc = DELTAS[best_move]
                nr, nc = (ar + dr) % rows, (ac + dc) % cols
                taken.discard((ar, ac))
                taken.add((nr, nc))
            else:
                taken.add((ar, ac))

        return out

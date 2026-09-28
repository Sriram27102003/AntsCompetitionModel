"""Grandmaster Tactical Teacher v4: Hill Raider Deathball Swarms & Deep Tournament Navigation."""
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

class TacticalTeacherV4:
    def __init__(self, hill_lead: int = 14, max_depth: int = 72):
        self.hill_lead = hill_lead
        self.max_depth = max_depth

    def compute_field(
        self,
        obs: dict,
        water: np.ndarray,
        seen: np.ndarray,
        target_hills: list[tuple[int, int]],
        target_food: list[tuple[int, int]],
        threat_mask: np.ndarray | None = None
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

        # 1. Enemy Hills: Absolute Top Priority (Level 0)
        if target_hills:
            hills_grid = np.zeros((rows, cols), dtype=bool)
            for r, c in target_hills:
                hills_grid[r, c] = True
            seed(hills_grid & passable, 0)

        # 2. Defense Threats near home hill (Level 2)
        if threat_mask is not None and threat_mask.any():
            seed(threat_mask & passable, 2)

        # 3. Food Economy (Level 12)
        if target_food:
            food_grid = np.zeros((rows, cols), dtype=bool)
            for r, c in target_food:
                food_grid[r, c] = True
            seed(food_grid & passable, self.hill_lead)

        # 4. Fog Frontier (Level 18)
        unseen = ~seen
        if unseen.any():
            seed(unseen & passable, self.hill_lead + 4)

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

        # Threat mask: enemy ants within 12 cells of our home hills
        threat_mask = np.zeros((rows, cols), dtype=bool)
        for er, ec, _ in foes:
            for hr, hc in my_hills:
                if wrap_dist2(er, ec, hr, hc, rows, cols) <= 144:
                    threat_mask[er, ec] = True
                    break

        food = [(f[0], f[1]) for f in obs.get("food", [])]
        dist = self.compute_field(obs, water, seen, enemy_hills, food, threat_mask)

        # Distance specifically to enemy hills
        hill_dist = np.full((rows, cols), FAR, dtype=np.int32)
        if enemy_hills:
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

        # Order of ant decision: Ants on home hill move FIRST
        def ant_priority(i: int):
            ar, ac = mine[i]
            on_hill = (ar, ac) in my_hills
            return (0 if on_hill else 1, dist[ar, ac], i)

        order = sorted(range(len(mine)), key=ant_priority)

        # Swarm mode active when colony has 15+ ants and an enemy hill is known
        siege_active = len(mine) >= 15 and len(enemy_hills) > 0

        for i in order:
            ar, ac = mine[i]
            on_my_hill = (ar, ac) in my_hills

            valid_moves = []
            for k in range(4):
                dr, dc = DELTAS[k]
                nr, nc = (ar + dr) % rows, (ac + dc) % cols

                # CRITICAL RULE: NEVER step onto a home hill unless an enemy is currently standing on it!
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

            # Is this ant within siege approach range (<= 30 cells from enemy hill)?
            near_enemy_hill = siege_active and (hill_dist[ar, ac] <= 30)

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

                # Combat logic:
                if is_razing_move:
                    combat_penalty = 25000.0  # Instant massive priority
                elif enemy_threats == 0:
                    combat_penalty = 0.0
                elif friendly_support >= enemy_threats:
                    combat_penalty = 500.0  # Advantageous 2v1 or 3v1 gank
                elif near_enemy_hill and hill_dist[dest_r, dest_c] < hill_dist[ar, ac]:
                    # Hill siege rush: Do NOT retreat if advancing toward enemy hill with local pack!
                    combat_penalty = -200.0  # Small friction, but pushes through defenders to raze
                else:
                    # In open field: avoid bad trades
                    combat_penalty = -5000.0

                target_d = dist[dest_r, dest_c]
                strategic_score = -float(target_d)

                # Extra siege draw towards enemy hill
                if near_enemy_hill:
                    h_d = hill_dist[dest_r, dest_c]
                    strategic_score += (30 - h_d) * 15.0

                # Massive urge to unblock home hill
                if on_my_hill and k != HOLD:
                    strategic_score += 1000.0

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

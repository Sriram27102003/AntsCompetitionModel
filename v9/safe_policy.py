"""A policy that never sends two of its own ants to one square, as part of the exported graph.

Every ant that ends a turn on a shared square dies, own ants included (engine `turn.rs`), and a
per-cell policy picks each ant's move alone, so it does this to itself: v10d collided on 6.4% of
turns. `Safe` wraps a memory model and rewrites only the logits of the ants that must give way, so
everything the network learned is kept:

  - a move into known water or onto food is a hold (the engine ignores it anyway);
  - where several ants would end on one square, an ant holding there keeps it, else the most
    confident mover does, and the rest hold;
  - a new hold can block someone else, so the rule is applied `rounds` times.

It is a fixed function of the board and the logits: no weights, so nothing to train.

    python safe_policy.py $PWD/runs/v10d/best.pt /abs/path/models/v10s_mini [--take-hills]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from tb_baselines import nets
from tb_baselines.export import classes, export
from tb_baselines.train.bc import override

STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))      # N E S W, the policy's first four channels
HOLD = 4
MINE, FOES, FOOD, WATER, FOE_HILL = 0, 1, 2, 3, 5   # board planes
BIG = 1.0e4
# enemies that can have a square in attack range (d2 <= 5) after one step of their own: d2 <= 10
REACH = [[float(dr * dr + dc * dc <= 10) for dc in range(-3, 4)] for dr in range(-3, 4)]


def at(x: torch.Tensor, d: int) -> torch.Tensor:
    """x read one step in direction d from each cell (the board wraps)."""
    dr, dc = STEPS[d]                                # one axis only: a roll by 0 still costs ops
    return torch.roll(x, shifts=-dr, dims=2) if dr else torch.roll(x, shifts=-dc, dims=3)


def to(x: torch.Tensor, d: int) -> torch.Tensor:
    """x carried one step in direction d."""
    dr, dc = STEPS[d]
    return torch.roll(x, shifts=dr, dims=2) if dr else torch.roll(x, shifts=dc, dims=3)


class Safe(nn.Module):
    memory_ports = True

    def __init__(self, inner: nn.Module, rounds: int = 16, take_hills: bool = False):
        super().__init__()
        # take_hills: an ant beside an enemy hill that no enemy can defend this turn steps onto it.
        # Razing is +2 and the only thing rank counts; v10e took 79% of such chances.
        self.take_hills = take_hills
        # The convolution passes give the same moves but measured slower in the platform's runtime
        # (3.4 ms a pass against about 2 on a laptop), so the rolled passes are the ones exported.
        self.slow = True
        self.register_buffer("reach", torch.tensor(REACH).reshape(1, 1, 7, 7), persistent=False)
        # One pass as two 3x3 convolutions instead of twelve rolls (used when `slow` is False).
        # `carry` moves each direction's movers (channels 0-3) and their scores (4-7) one step on:
        # output 0 counts arrivals, outputs 1-4 are the score arriving from each direction.
        # `back` reads a square's verdict from the square each direction's mover started on.
        carry = torch.zeros(5, 8, 3, 3)
        back = torch.zeros(4, 4, 3, 3)
        for d, (dr, dc) in enumerate(STEPS):
            carry[0, d, 1 - dr, 1 - dc] = 1.0
            carry[1 + d, 4 + d, 1 - dr, 1 - dc] = 1.0
            back[d, d, 1 + dr, 1 + dc] = 1.0
        self.register_buffer("carry", carry, persistent=False)
        self.register_buffer("back", back, persistent=False)
        self.register_buffer("dirs", torch.arange(4).reshape(1, 4, 1, 1), persistent=False)
        self.inner = inner
        self.trunk = getattr(inner, "trunk", inner)      # export.shape_of reads the architecture here
        self.channels = inner.channels
        self.memory_kind = inner.memory_kind
        self.rounds = rounds

    def fix(self, board: torch.Tensor, policy: torch.Tensor) -> torch.Tensor:
        mine = board[:, MINE:MINE + 1] > 0
        blocked = (board[:, FOOD:FOOD + 1] + board[:, WATER:WATER + 1]) > 0
        best = policy.argmax(1, keepdim=True)
        conf = policy.max(1, keepdim=True).values
        move = [mine & (best == d) for d in range(4)]
        if self.take_hills:
            foes = board[:, FOES:FOES + 1].to(policy.dtype)
            threat = torch.nn.functional.conv2d(
                torch.nn.functional.pad(foes, (3, 3, 3, 3), mode="circular"), self.reach.to(policy.dtype)) > 0.5
            free = (board[:, FOE_HILL:FOE_HILL + 1] > 0) & ~threat
            want = [mine & at(free, d) for d in range(4)]
            forced = want[0] | want[1] | want[2] | want[3]
            for d in (1, 2, 3):                               # one direction an ant: the first
                want[d] = want[d] & ~(want[0] if d == 1 else want[0] | want[1] if d == 2
                                      else want[0] | want[1] | want[2])
            move = [(m & ~forced) | w for m, w in zip(move, want)]
            for d in range(4):
                conf = torch.where(want[d], torch.full_like(conf, BIG - d), conf)
        move = [m & ~at(blocked, d) for d, m in enumerate(move)]
        if not self.slow:
            pad = lambda x: torch.nn.functional.pad(x, (1, 1, 1, 1), mode="circular")
            dt = policy.dtype
            mine_f = mine.to(dt)
            mv = torch.cat(move, dim=1).to(dt)                         # [B, 4, H, W], 0/1
            for _ in range(self.rounds):
                hold = mine_f - mv.sum(1, keepdim=True)
                got = torch.nn.functional.conv2d(
                    pad(torch.cat([mv, mv * conf + (mv - 1.0) * BIG], dim=1)), self.carry.to(dt))
                conflict = (got[:, :1] + hold) > 1.5
                win = got[:, 1:].argmax(1, keepdim=True)
                lose = (conflict & ((hold > 0.5) | (win != self.dirs))).to(dt)
                mv = mv * (1.0 - torch.nn.functional.conv2d(pad(lose), self.back.to(dt)))
            hold = mine_f - mv.sum(1, keepdim=True)
            return policy + torch.cat([mv, hold], dim=1) * BIG
        for _ in range(self.rounds):
            hold = mine & ~(move[0] | move[1] | move[2] | move[3])
            arrive = hold.float() + sum(to(m.float(), d) for d, m in enumerate(move))
            conflict = arrive > 1.5
            # the mover each square would keep: the most confident one arriving
            score = torch.cat([to(torch.where(m, conf, torch.full_like(conf, -BIG)), d)
                               for d, m in enumerate(move)], dim=1)
            win = score.argmax(1, keepdim=True)
            move = [m & ~at(conflict & (hold | (win != d)), d) for d, m in enumerate(move)]
        hold = mine & ~(move[0] | move[1] | move[2] | move[3])
        final = torch.cat(move + [hold], dim=1).to(policy.dtype)   # one-hot of each ant's move
        return policy + final * BIG

    def forward(self, board: torch.Tensor, memory_in: torch.Tensor):
        policy, memory = self.inner(board, memory_in)
        return self.fix(board, policy), memory


def load(weights: Path, rounds: int = 16, take_hills: bool = False) -> Safe:
    state = torch.load(weights, map_location="cpu", weights_only=False)["trunk"]

    class A:
        arch, blocks = "mini_trunk", 6
        channels = next(v for k, v in state.items() if k.endswith("mix.weight")).shape[0]
    inner = nets.build(override(classes()["mini"], A), memory=True)
    inner.load_state_dict(state)
    return Safe(inner, rounds, take_hills).eval()


if __name__ == "__main__":
    weights, out = Path(sys.argv[1]), Path(sys.argv[2])
    hills = "--take-hills" in sys.argv[3:]
    # the weight class is measured from the file; every class plays the same Open ladder
    cls = "small" if "--small" in sys.argv[3:] else "mini"
    rounds = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 16
    m = export(load(weights, rounds=rounds, take_hills=hills), cls, out,
               f"FatAss {out.name}: {weights} with own-ant collisions removed in the graph"
               + (" and undefended adjacent enemy hills taken" if hills else "") + " (safe_policy.py)",
               f"python safe_policy.py {weights} {out}" + (" --take-hills" if hills else "")
               + f" --rounds {rounds}" + (" --small" if cls == "small" else ""))
    print(f"S = {m['size_metric_bytes']:,} bytes, inference {m['infer_us_max'] / 1000:.1f} ms -> {out}")

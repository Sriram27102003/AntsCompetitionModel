"""Encode a JSONL dataset once into bit-packed arrays, so training never parses JSON again.

Every board plane and memory plane is 0/1, so a row is 9 bits a cell: about 3 KB for a 48x64
board. Output is one .npz per input file, grouped by board size:

    python prep_fast.py data/strong-11.jsonl.gz data/strong-relabel.jsonl.gz --out packed/
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tb_baselines.planes import MOVES, encode, encode_memory

MOVE_INDEX = {m: i for i, m in enumerate(MOVES)}


def pack(args) -> str:
    src, out, limit = args
    groups = defaultdict(lambda: {"x": [], "ant_row": [], "ant_r": [], "ant_c": [], "ant_y": []})
    n = 0
    with gzip.open(src, "rt") as f:
        f.readline()                                   # header
        for line in f:
            r = json.loads(line)
            o = r["o"]
            if not o["mine"] or len(r["a"]) != len(o["mine"]):
                continue
            size = tuple(o["size"])
            x = np.concatenate([encode(o)[0], encode_memory(r.get("m") or None, size)[0]], 0)
            g = groups[size]
            row = len(g["x"])
            g["x"].append(np.packbits(x.astype(np.uint8).ravel()))
            for (ar, ac), ch in zip(o["mine"], r["a"]):
                g["ant_row"].append(row)
                g["ant_r"].append(ar)
                g["ant_c"].append(ac)
                g["ant_y"].append(MOVE_INDEX.get(ch, MOVE_INDEX["-"]))
            n += 1
            if limit and n >= limit:
                break
    arrays = {}
    for (h, w), g in groups.items():
        k = f"{h}x{w}"
        arrays[f"{k}_x"] = np.stack(g["x"])
        arrays[f"{k}_ant_row"] = np.array(g["ant_row"], dtype=np.int32)
        arrays[f"{k}_ant_r"] = np.array(g["ant_r"], dtype=np.int16)
        arrays[f"{k}_ant_c"] = np.array(g["ant_c"], dtype=np.int16)
        arrays[f"{k}_ant_y"] = np.array(g["ant_y"], dtype=np.int8)
    np.savez(out, **arrays)
    return f"{src.name}: {n:,} rows -> {out.name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="rows per input, 0 for all")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, a.out / (p.name.replace(".jsonl.gz", "") + ".npz"), a.limit) for p in a.inputs]
    with Pool(min(len(jobs), 4)) as pool:
        for msg in pool.imap_unordered(pack, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()

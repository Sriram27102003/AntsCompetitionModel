"""Download every Hive Minds replay on QA, with its match record, into replays_qa/.

Safe to re-run: files already on disk are skipped, so each run adds only the matches played since.

    python fetch_replays.py [--season hive-minds]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

API = "https://qa.tinybrains.dev/v1"
OUT = Path(__file__).parent / "replays_qa"


def get(url: str):
    return json.load(urllib.request.urlopen(url, timeout=60))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="hive-minds")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    ms, cur = [], None
    while True:
        q = f"{API}/matches?game=ants&season={a.season}&limit=100"
        d = get(q + (f"&cursor={urllib.parse.quote(cur)}" if cur else ""))
        ms += d["matches"]
        cur = d.get("next_cursor")
        if not cur:
            break

    def one(m: dict) -> str:
        p = OUT / f"{m['id']}.replay.json"
        if p.exists():
            return "cached"
        rec = get(f"{API}/matches/{m['id']}")
        if not rec.get("replay_url"):
            return "no-replay"
        p.write_bytes(urllib.request.urlopen(rec["replay_url"], timeout=120).read())
        (OUT / f"{m['id']}.match.json").write_text(json.dumps(rec))
        return "new"

    with cf.ThreadPoolExecutor(8) as ex:
        res = Counter(ex.map(one, ms))
    seats = Counter(s["model"] for m in ms for s in m["seats"])
    print(f"{len(ms)} matches in {a.season}: {dict(res)}")
    print("seats by model:", dict(seats.most_common()))


if __name__ == "__main__":
    main()

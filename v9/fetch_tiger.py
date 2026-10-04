"""Download every public Hive Minds replay with TigerAnts or FireAnts in it, plus its match record."""
import json, urllib.request, urllib.parse, concurrent.futures as cf
from pathlib import Path
API = "https://qa.tinybrains.dev/v1"
OUT = Path(__file__).parent / "tiger"
get = lambda u: json.load(urllib.request.urlopen(u, timeout=60))
ms, cur = [], None
while True:
    d = get(f"{API}/matches?game=ants&limit=100" + (f"&cursor={urllib.parse.quote(cur)}" if cur else ""))
    ms += d["matches"]; cur = d.get("next_cursor")
    if not cur: break
tg = [m for m in ms if any(s["model"] in ("TigerAnts", "FireAnts") for s in m["seats"])]
def one(m):
    p = OUT / f"{m['id']}.replay.json"
    if p.exists(): return "cached"
    rec = get(f"{API}/matches/{m['id']}")
    (OUT / f"{m['id']}.match.json").write_text(json.dumps(rec))
    if not rec.get("replay_url"): return "no-replay"
    p.write_bytes(urllib.request.urlopen(rec["replay_url"], timeout=120).read())
    return "ok"
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, tg))
from collections import Counter
print(len(ms), "matches;", len(tg), "with TigerAnts/FireAnts;", Counter(res))

import json
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "FatAss-v6": {
        "weights": str(PROJECT_ROOT / "models" / "v6_mini" / "model.onnx"),
        "manifest": str(PROJECT_ROOT / "models" / "v6_mini" / "manifest.json"),
    },
    "FatAss-v5": {
        "weights": str(PROJECT_ROOT / "models" / "v5_mini" / "model.onnx"),
        "manifest": str(PROJECT_ROOT / "models" / "v5_mini" / "manifest.json"),
    },
    "FatAss-v2": {
        "weights": str(PROJECT_ROOT / "models" / "championship_mini" / "model.onnx"),
        "manifest": str(PROJECT_ROOT / "models" / "championship_mini" / "manifest.json"),
    }
}

MATCHES_3P = [
    {"id": "3p-v2-v5-v6-basic", "map": "basic-small-3p", "seats": ["FatAss-v6", "FatAss-v5", "FatAss-v2"]},
    {"id": "3p-v2-v5-v6-cave", "map": "season_maps/small-cave-3p-1h.json", "seats": ["FatAss-v6", "FatAss-v5", "FatAss-v2"]},
    {"id": "3p-v2-v5-v6-open", "map": "season_maps/medium-open-3p-2h.json", "seats": ["FatAss-v6", "FatAss-v5", "FatAss-v2"]},
    {"id": "3p-v2-v5-v6-rooms", "map": "season_maps/large-rooms-3p-3h.json", "seats": ["FatAss-v6", "FatAss-v5", "FatAss-v2"]},
]

MATCHES_DUELS = [
    # v6 vs v2
    {"id": "duel-v6-vs-v2-tiny", "map": "basic-tiny-2p", "seats": ["FatAss-v6", "FatAss-v2"]},
    {"id": "duel-v6-vs-v2-rooms", "map": "season_maps/tiny-rooms-2p-1h.json", "seats": ["FatAss-v6", "FatAss-v2"]},
    # v6 vs v5
    {"id": "duel-v6-vs-v5-tiny", "map": "basic-tiny-2p", "seats": ["FatAss-v6", "FatAss-v5"]},
    {"id": "duel-v6-vs-v5-rooms", "map": "season_maps/tiny-rooms-2p-1h.json", "seats": ["FatAss-v6", "FatAss-v5"]},
    # v5 vs v2
    {"id": "duel-v5-vs-v2-tiny", "map": "basic-tiny-2p", "seats": ["FatAss-v5", "FatAss-v2"]},
    {"id": "duel-v5-vs-v2-rooms", "map": "season_maps/tiny-rooms-2p-1h.json", "seats": ["FatAss-v5", "FatAss-v2"]},
]

def run_suite(name, matches, out_dir):
    rows = []
    for m in matches:
        seat_entries = []
        for seat_idx, bot_name in enumerate(m["seats"]):
            cfg = MODELS[bot_name]
            seat_entries.append({
                "seat": seat_idx,
                "weights": cfg["weights"],
                "manifest": cfg["manifest"],
                "label": bot_name
            })
        rows.append({
            "id": m["id"],
            "seed": 42,
            "map": m["map"],
            "seat_count": len(m["seats"]),
            "seats": seat_entries
        })
    doc = {"game": "ants", "vars": {"max_turns": 350}, "rows": rows}
    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    fpath = PROJECT_ROOT / f"suite_{slug}.json"
    fpath.write_text(json.dumps(doc, indent=2))

    print(f"\n{'='*75}\n[RUNNING {name.upper()}]\n{'='*75}", flush=True)
    cmd = ["tinybrains", "run", str(fpath), "--out", str(out_dir)]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    print(res.stdout)
    if res.stderr:
        print("[STDERR]", res.stderr)

    results = []
    for m in matches:
        rep_file = out_dir / f"{m['id']}.json"
        if not rep_file.exists():
            continue
        with open(rep_file) as f:
            d = json.load(f)
        scores = d.get("scores", [])
        seat_results = []
        for idx, bot in enumerate(m["seats"]):
            score = scores[idx] if idx < len(scores) else 0
            seat_results.append((bot, score))
        seat_results.sort(key=lambda x: -x[1])
        winner = seat_results[0][0]
        results.append({
            "id": m["id"],
            "map": m["map"],
            "winner": winner,
            "results": seat_results,
            "turns": d.get("turns") or len(d.get("deltas", []))
        })
    return results

def main():
    out_dir = PROJECT_ROOT / "comparison_v2_v5_v6_replays"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("  HEAD-TO-HEAD TITAN SHOWDOWN: FatAss v2 vs FatAss v5 vs FatAss v6")
    print("="*80, flush=True)

    results_3p = run_suite("3-Player Titan Showdown (v6 vs v5 vs v2)", MATCHES_3P, out_dir)
    results_duels = run_suite("Head-to-Head Duels (v6 vs v5 vs v2)", MATCHES_DUELS, out_dir)

    print("\n" + "="*95)
    print("FINAL COMPARISON SCORECARD: FatAss v2 vs FatAss v5 vs FatAss v6")
    print("="*95)
    print(f"{'Match ID':<24} | {'Map':<26} | {'Winner':<12} | {'Final Scores / Standings'}")
    print("-" * 95)
    for r in results_3p:
        scores_str = ", ".join([f"{b}: {s}pts" for b, s in r["results"]])
        print(f"{r['id']:<24} | {r['map']:<26} | {r['winner']:<12} | {scores_str}")
    print("-" * 95)
    for r in results_duels:
        scores_str = ", ".join([f"{b}: {s}pts" for b, s in r["results"]])
        print(f"{r['id']:<24} | {r['map']:<26} | {r['winner']:<12} | {scores_str}")
    print("=" * 95, flush=True)

if __name__ == "__main__":
    main()

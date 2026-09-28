import json
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "FatAss-v7": {
        "weights": str(PROJECT_ROOT / "models" / "v7_mini" / "model.onnx"),
        "manifest": str(PROJECT_ROOT / "models" / "v7_mini" / "manifest.json"),
    },
    "FatAss-v6": {
        "weights": str(PROJECT_ROOT / "models" / "v6_mini" / "model.onnx"),
        "manifest": str(PROJECT_ROOT / "models" / "v6_mini" / "manifest.json"),
    },
    "FatAss-v5": {
        "weights": str(PROJECT_ROOT / "models" / "v5_mini" / "model.onnx"),
        "manifest": str(PROJECT_ROOT / "models" / "v5_mini" / "manifest.json"),
    },
    "FatAss-v4": {
        "weights": str(PROJECT_ROOT / "models" / "v4_mini" / "model.onnx"),
        "manifest": str(PROJECT_ROOT / "models" / "v4_mini" / "manifest.json"),
    },
    "FatAss-v3": {
        "weights": str(PROJECT_ROOT / "models" / "v3_mini" / "model.onnx"),
        "manifest": str(PROJECT_ROOT / "models" / "v3_mini" / "manifest.json"),
    },
    "FatAss-v2": {
        "weights": str(PROJECT_ROOT / "models" / "championship_mini" / "model.onnx"),
        "manifest": str(PROJECT_ROOT / "models" / "championship_mini" / "manifest.json"),
    },
    "micro-bc": {
        "weights": str(PROJECT_ROOT / "ants-starter" / "models" / "micro-bc" / "model.onnx"),
        "manifest": str(PROJECT_ROOT / "ants-starter" / "models" / "micro-bc" / "manifest.json"),
    }
}

SUITES = {
    "2-Player Duels": [
        {"id": "1v1-v7-vs-v5-tiny", "map": "basic-tiny-2p", "seats": ["FatAss-v7", "FatAss-v5"]},
        {"id": "1v1-v7-vs-v5-rooms", "map": "season_maps/tiny-rooms-2p-1h.json", "seats": ["FatAss-v7", "FatAss-v5"]},
        {"id": "1v1-v7-vs-v2-tiny", "map": "basic-tiny-2p", "seats": ["FatAss-v7", "FatAss-v2"]},
    ],
    "4-Player FFA Championship": [
        {"id": "4p-ffa-medium", "map": "basic-medium-4p", "seats": ["FatAss-v7", "FatAss-v5", "FatAss-v2", "micro-bc"]},
        {"id": "4p-ffa-cave", "map": "season_maps/small-cave-4p-2h.json", "seats": ["FatAss-v7", "FatAss-v5", "FatAss-v2", "micro-bc"]},
        {"id": "4p-ffa-open", "map": "season_maps/tiny-open-4p-1h.json", "seats": ["FatAss-v7", "FatAss-v5", "FatAss-v2", "micro-bc"]},
    ]
}




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
    doc = {"game": "ants", "vars": {"max_turns": 300}, "rows": rows}
    slug = name.lower().replace(" ", "_")
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

    print(f"{'='*75}\nRESULTS: {name.upper()}\n{'='*75}")
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
        results_str = ", ".join([f"{b}: {s}pts" for b, s in seat_results])
        winner = seat_results[0][0]
        print(f"  {m['id']:<24} | Map: {m['map']:<28} | Winner: {winner:<10} | {results_str}")

def main():
    out_dir = PROJECT_ROOT / "benchmark_replays"
    out_dir.mkdir(parents=True, exist_ok=True)
    for suite_name, matches in SUITES.items():
        run_suite(suite_name, matches, out_dir)



if __name__ == "__main__":
    main()

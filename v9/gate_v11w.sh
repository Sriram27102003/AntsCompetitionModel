#!/bin/bash
# After the wide (96-channel) training ends: export it, check it, gate it in the Python harness
# against v10e, the teacher and the TigerAnts clone, then play the exported files through the real CLI.
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sriram/Documents/tinybrains-ants/baselines/src:$PWD
export TINYBRAINS=$PWD/bin/tinybrains
PY=../.venv/bin/python
while pgrep -f "train_fast.py packed" >/dev/null; do sleep 30; done
[ -f runs/v11w/best.pt ] || { echo "no v11w checkpoint"; exit 1; }
echo "[$(date +%H:%M)] exporting"
$PY safe_policy.py $PWD/runs/v11w/best.pt /Users/sriram/Documents/AntsCompitionModel/models/v11w_small --take-hills --small 2>&1 | tail -2
$TINYBRAINS check ../models/v11w_small/model.onnx ../models/v11w_small/manifest.json --memory-flat-bytes 102400 --memory-cell-bytes 1 2>&1 | grep -E "PASSED|FAILED|median|round trip"
echo "[$(date +%H:%M)] gate"
for M in "*-2p-*" "*-4p-*" "*-6p-*"; do
  echo "=== $M"
  $PY -u duel.py hill:runs/v11w/best.pt hill:runs/v10e/best.pt strong ckpt:runs/clone-TigerAnts/best.pt --maps "$M" --games 96 --max-turns 1000 --parallel 12 --seed 6100 2>&1 | grep -v "not the referee\|^ants on\|^$\|seat-turns\|^\["
done > gate_v11w.log
echo "[$(date +%H:%M)] arena"
$PY arena.py ../models/v11w_small ../models/v10s_mini ../models/v10f_mini --per-size 3 --seeds 11 --tag v11w > arena_v11w.log 2>&1
echo "[$(date +%H:%M)] GATE_V11W_DONE"

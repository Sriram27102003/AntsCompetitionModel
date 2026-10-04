#!/bin/bash
# Step 2 -> 3: strong-teacher data -> v10 -> export -> admission check -> head-to-head gate.
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sriram/Documents/tinybrains-ants/baselines/src
export TINYBRAINS=/Users/sriram/Documents/AntsCompitionModel/v9/bin/tinybrains
PY=../.venv/bin/python
while pgrep -f collect_strong.py >/dev/null; do sleep 20; done
echo "[$(date +%H:%M)] packing"
$PY prep_fast.py data/strong-11.jsonl.gz data/strong-12.jsonl.gz data/strong-relabel.jsonl.gz --out packed || exit 1
echo "[$(date +%H:%M)] training v10"
$PY -u train_fast.py packed/*.npz --init ../runs/v5_mini/best.pt --epochs 2 --lr 1e-3 --out runs/v10 > train_v10.log 2>&1 || exit 1
echo "[$(date +%H:%M)] exporting"
$PY -m tb_baselines.export --class mini --arch mini_trunk --channels 64 --blocks 6 --memory max \
  --weights runs/v10/best.pt --out ../models/v10_mini --method "FatAss v10: strong-teacher BC (self-play + relabelled QA ladder positions), MiniTrunk(64,6) + max memory, warm start v5" > export_v10.log 2>&1
$TINYBRAINS check ../models/v10_mini/model.onnx ../models/v10_mini/manifest.json --memory-flat-bytes 102400 --memory-cell-bytes 1 > check_v10.log 2>&1
echo "[$(date +%H:%M)] gate"
for M in "tiny*-2p-* small*-2p-* medium*-2p-*" "*-4p-*" "*-6p-*"; do
  echo "=== $M"
  $PY -u duel.py ckpt:runs/v10/best.pt ckpt:../runs/v5_mini/best.pt ckpt:runs/clone-TigerAnts/best.pt ckpt:runs/clone-FireAnts/best.pt strong --maps $M --games 60 --max-turns 500 --parallel 16 2>&1 | grep -v "not the referee\|^ants on\|^$\|seat-turns\|^\["
done > gate_v10.log
echo "[$(date +%H:%M)] PIPELINE_DONE"

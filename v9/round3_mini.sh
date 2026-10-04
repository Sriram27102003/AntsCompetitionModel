#!/bin/bash
# Round 3, mini-sized: once the dagger3 collectors finish, pack, then train the 94-channel trim of
# v11w (runs/v11m/init.pt, fits the mini class) two epochs on every shard -> runs/v11n; then gate.
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sriram/Documents/tinybrains-ants/baselines/src
export TINYBRAINS=$PWD/bin/tinybrains
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6
PY=../.venv/bin/python
while pgrep -f "collect_strong.py dagger" >/dev/null; do sleep 30; done
echo "[$(date +%H:%M)] packing"
$PY prep_fast.py data/dagger3-self.jsonl.gz data/dagger3-mixed.jsonl.gz --out packed || exit 1
echo "[$(date +%H:%M)] training v11n"
$PY -u train_fast.py packed/*.npz --init runs/v11m/init.pt --channels 94 --batch 32 --epochs 2 --lr 2.5e-4 --fight-weight 2 --out runs/v11n > train_v11n.log 2>&1 || exit 1
echo "[$(date +%H:%M)] gate"
for M in "*-2p-*" "*-4p-*" "*-6p-*"; do
  echo "=== $M"
  $PY -u duel.py hill:runs/v11n/best.pt hill:runs/v11w/best.pt strong ckpt:runs/clone-MadPanda/best.pt --maps "$M" --games 96 --max-turns 1000 --parallel 12 --seed 8100 2>&1 | grep -v "not the referee\|^ants on\|^$\|seat-turns\|^\["
done > gate_v11n.log
echo "[$(date +%H:%M)] ROUND3_MINI_DONE"

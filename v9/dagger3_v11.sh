#!/bin/bash
# Round 3: the wide net (v11w) plays as exported, the strong teacher labels its states; it is then
# trained two more epochs on every shard, old ones included -> runs/v11x. Gate at the end; nothing
# is exported or replaced here. data/ and packed/ live on /Volumes/Thaniska.
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sriram/Documents/tinybrains-ants/baselines/src
export TINYBRAINS=$PWD/bin/tinybrains
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6
PY=../.venv/bin/python
S=runs/v11w/best.pt
[ -d /Volumes/Thaniska/AntsCompitionModel/v9/packed ] || { echo "external drive not mounted"; exit 1; }
echo "[$(date +%H:%M)] collecting"
$PY -u collect_strong.py dagger --hill --student $S --seat-turns 60000 --max-turns 1000 --seed 41 --out data/dagger3-self.jsonl.gz > dagger3_self.log 2>&1 &
$PY -u collect_strong.py dagger --hill --student $S --opponents strong ckpt:runs/clone-MadPanda/best.pt hill:runs/v10e/best.pt ckpt:runs/clone-TigerAnts/best.pt --seat-turns 50000 --max-turns 1000 --seed 42 --out data/dagger3-mixed.jsonl.gz > dagger3_mixed.log 2>&1 &
wait
echo "[$(date +%H:%M)] packing"
$PY prep_fast.py data/dagger3-self.jsonl.gz data/dagger3-mixed.jsonl.gz --out packed || exit 1
echo "[$(date +%H:%M)] training v11x"
$PY -u train_fast.py packed/*.npz --init $S --channels 96 --batch 32 --epochs 2 --lr 2.5e-4 --fight-weight 2 --out runs/v11x > train_v11x.log 2>&1 || exit 1
echo "[$(date +%H:%M)] gate"
for M in "*-2p-*" "*-4p-*" "*-6p-*"; do
  echo "=== $M"
  $PY -u duel.py hill:runs/v11x/best.pt hill:$S strong ckpt:runs/clone-MadPanda/best.pt --maps "$M" --games 96 --max-turns 1000 --parallel 12 --seed 8100 2>&1 | grep -v "not the referee\|^ants on\|^$\|seat-turns\|^\["
done > gate_v11x.log
echo "[$(date +%H:%M)] DAGGER3_DONE"

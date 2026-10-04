#!/bin/bash
# DAgger round 1: v10 plays, the strong teacher labels its states, v10 is retrained -> runs/v10d.
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sriram/Documents/tinybrains-ants/baselines/src
export TINYBRAINS=$PWD/bin/tinybrains
PY=../.venv/bin/python
S=runs/v10/epoch1_backup.pt
echo "[$(date +%H:%M)] collecting"
$PY -u collect_strong.py dagger --student $S --seat-turns 75000 --seed 21 --out data/dagger1-self.jsonl.gz > dagger1_self.log 2>&1 &
$PY -u collect_strong.py dagger --student $S --opponents ckpt:runs/clone-TigerAnts/best.pt strong ckpt:runs/clone-FireAnts/best.pt --seat-turns 40000 --seed 22 --out data/dagger1-mixed.jsonl.gz > dagger1_mixed.log 2>&1 &
wait
echo "[$(date +%H:%M)] packing"
$PY prep_fast.py data/dagger1-self.jsonl.gz data/dagger1-mixed.jsonl.gz --out packed || exit 1
echo "[$(date +%H:%M)] training v10d"
$PY -u train_fast.py packed/*.npz --init $S --epochs 2 --lr 5e-4 --out runs/v10d > train_v10d.log 2>&1 || exit 1
echo "[$(date +%H:%M)] gate"
for M in "tiny*-2p-* small*-2p-* medium*-2p-*" "*-4p-*" "*-6p-*"; do
  echo "=== $M"
  $PY -u duel.py ckpt:runs/v10d/best.pt ckpt:$S ckpt:../runs/v5_mini/best.pt ckpt:runs/clone-TigerAnts/best.pt strong --maps $M --games 60 --max-turns 500 --parallel 16 2>&1 | grep -v "not the referee\|^ants on\|^$\|seat-turns\|^\["
done > gate_v10d.log
echo "[$(date +%H:%M)] DAGGER_DONE"

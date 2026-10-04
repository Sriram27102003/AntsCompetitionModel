#!/bin/bash
# Waits for the remembering-teacher dataset, then trains v9a (warm start from v5) and v9b (scratch).
cd "$(dirname "$0")"
while pgrep -f tb_baselines.collect >/dev/null; do sleep 15; done
export PYTHONPATH=/Users/sriram/Documents/tinybrains-ants/baselines/src
export TINYBRAINS=/Users/sriram/Documents/AntsCompitionModel/v9/bin/tinybrains
PY=../.venv/bin/python
COMMON="--class mini --arch mini_trunk --channels 64 --blocks 6 --memory --data data/hive-remember-300k.jsonl.gz"
$PY -m tb_baselines.train.bc $COMMON --init ../runs/v5_mini/best.pt --lr 5e-4 --epochs 6 --seed 1 --out runs/v9a > train_v9a.log 2>&1 &
$PY -m tb_baselines.train.bc $COMMON --lr 2e-3 --epochs 8 --seed 2 --out runs/v9b > train_v9b.log 2>&1 &
wait
echo TRAINING_DONE

#!/bin/bash
# Sparring partners: behaviour clones of FireAnts and TigerAnts from their recorded QA games.
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sriram/Documents/tinybrains-ants/baselines/src
for who in FireAnts TigerAnts; do
  ../.venv/bin/python -u -m tb_baselines.train.bc --class mini --arch mini_trunk --channels 64 --blocks 6 \
    --memory --data data/clone-$who.jsonl.gz --init ../runs/v5_mini/best.pt \
    --lr 1e-3 --batch 64 --epochs 8 --holdout 0.1 --seed 4 --out runs/clone-$who > train_clone-$who.log 2>&1
done
echo CLONES_DONE >> train_clone-TigerAnts.log

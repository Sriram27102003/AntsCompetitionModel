#!/bin/bash
# After v9a finishes, fine-tune it on TigerAnts' own recorded moves (v9t).
cd "$(dirname "$0")"
while kill -0 5187 2>/dev/null; do sleep 30; done
export PYTHONPATH=/Users/sriram/Documents/tinybrains-ants/baselines/src
../.venv/bin/python -u -m tb_baselines.train.bc --class mini --arch mini_trunk --channels 64 --blocks 6 \
  --memory --data data/tiger-only.jsonl.gz --init runs/v9a/best.pt \
  --lr 4e-4 --batch 128 --epochs 5 --holdout 0.1 --seed 3 --out runs/v9t > train_v9t.log 2>&1
echo FINETUNE_DONE >> train_v9t.log

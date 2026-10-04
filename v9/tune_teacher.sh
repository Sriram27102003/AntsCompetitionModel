#!/bin/bash
# Pairwise teacher tournament: each candidate setting against the baseline teacher, 160 two-player
# games each, so every game is that pairing. Reads as head to head plus points (razed / lost).
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sriram/Documents/tinybrains-ants/baselines/src
export TINYBRAINS=$PWD/bin/tinybrains
BASE="strong:take_hill=1"
run() {  # $1 = policy A, $2 = policy B, $3 = maps, $4 = games, $5 = seed
  echo "##### $1  vs  $2   [$3]"
  ../.venv/bin/python -u duel.py "$1" "$2" --maps "$3" --games $4 --max-turns 1000 --parallel 12 --seed $5 2>&1 \
    | grep -E "score|vs|games in"
}
run strong "$BASE" "*-2p-*" 160 4000
for C in loss_behind=1.0 loss_behind=1.3 loss_behind=2.2 loss_ahead=0.7 guard_radius2=60 guard_radius2=12 \
         fight_radius2=29 passes=3 raze_min_ants=4 food_depth=60; do
  run "$BASE" "strong:take_hill=1,$C" "*-2p-*" 160 4000
done
echo TUNE_DONE

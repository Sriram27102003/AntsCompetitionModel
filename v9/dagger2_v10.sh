#!/bin/bash
# DAgger round 2: v10d plays as it is exported (own-ant collisions removed), the strong teacher labels
# its states; the champions' production positions are relabelled too; v10d is retrained with every
# earlier shard kept and fights weighted -> runs/v10e. data/ and packed/ live on /Volumes/Thaniska.
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sriram/Documents/tinybrains-ants/baselines/src
export TINYBRAINS=$PWD/bin/tinybrains
PY=../.venv/bin/python
S=runs/v10d/best.pt
[ -d /Volumes/Thaniska/AntsCompitionModel/v9/packed ] || { echo "external drive not mounted"; exit 1; }
echo "[$(date +%H:%M)] collecting"
$PY -u collect_strong.py dagger --safe --student $S --seat-turns 60000 --max-turns 1000 --seed 31 --out data/dagger2-self.jsonl.gz > dagger2_self.log 2>&1 &
$PY -u collect_strong.py dagger --safe --student $S --opponents strong ckpt:runs/clone-TigerAnts/best.pt safe:runs/clone-Anxiety/best.pt ckpt:runs/clone-FireAnts/best.pt --seat-turns 50000 --max-turns 1000 --seed 32 --out data/dagger2-mixed.jsonl.gz > dagger2_mixed.log 2>&1 &
wait
echo "[$(date +%H:%M)] relabelling champion positions"
$PY - <<'PYEOF' || exit 1
import gzip
with gzip.open("data/champ-top2.jsonl.gz", "wt") as out:
    for i, name in enumerate(("data/champ-anxiety.jsonl.gz", "data/champ-riot.jsonl.gz")):
        with gzip.open(name, "rt") as f:
            head = f.readline()
            if i == 0:
                out.write(head)
            for line in f:
                out.write(line)
PYEOF
$PY -u collect_strong.py relabel data/champ-top2.jsonl.gz --out data/strong-champ.jsonl.gz > relabel_champ.log 2>&1 || exit 1
rm -f data/champ-top2.jsonl.gz
echo "[$(date +%H:%M)] packing"
$PY prep_fast.py data/dagger2-self.jsonl.gz data/dagger2-mixed.jsonl.gz data/strong-champ.jsonl.gz --out packed || exit 1
echo "[$(date +%H:%M)] training v10e"
$PY -u train_fast.py packed/*.npz --init $S --epochs 2 --lr 5e-4 --fight-weight 2 --out runs/v10e > train_v10e.log 2>&1 || exit 1
echo "[$(date +%H:%M)] gate"
for M in "*-2p-*" "*-4p-*" "*-6p-*"; do
  echo "=== $M"
  $PY -u duel.py safe:runs/v10e/best.pt safe:$S strong ckpt:runs/clone-TigerAnts/best.pt --maps $M --games 60 --max-turns 1000 --parallel 12 --seed 2100 2>&1 | grep -v "not the referee\|^ants on\|^$\|seat-turns\|^\["
done > gate_v10e.log
echo "[$(date +%H:%M)] DAGGER2_DONE"

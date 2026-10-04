# FatAss II — `v11h_small`

The 96-channel entry: `models/v11h_small/` (`model.onnx` + `manifest.json`), 1,063,636 bytes, weight class **small**.
On qa.tinybrains.dev (Hive Minds) it played as **FatAss II v1**: 195 wins in 200 games, no strikes.

## What it is

- **Network:** MiniTrunk, 9 input planes (7 board + 2 memory), seven 3x3 convolutions at dilations
  1, 2, 4, 4, 2, 1, 1 and a 1x1 mix, 96 channels, 516,085 parameters. Weights: `v9/runs/v11w/best.pt`.
- **Training:** behaviour cloning of `v9/strong_teacher.py` with DAgger rounds, warm-started
  v5 -> v10 -> v10d -> v10e (64 channels), then widened to 96 channels with a function-preserving
  warm start and trained two epochs on every earlier dataset (`v9/train_v11w.log`).
- **In-graph rules (`v9/safe_policy.py`, no weights):** own-ant collisions are removed (8 passes),
  and an ant beside an enemy hill no enemy can defend that turn steps onto it.

## Rebuild it

```sh
cd v9
export PYTHONPATH=<path to Tiny-Brains/ants>/baselines/src:$PWD TINYBRAINS=<path to tinybrains CLI>
# train (data shards are not in this repository)
python train_fast.py packed/*.npz --init runs/v10e/best.pt --channels 96 --batch 32 --epochs 2 --lr 4e-4 --fight-weight 2 --out runs/v11w
# export and check
python safe_policy.py $PWD/runs/v11w/best.pt $PWD/../models/v11h_small --take-hills --rounds 8 --small
tinybrains check ../models/v11h_small/model.onnx ../models/v11h_small/manifest.json --memory-flat-bytes 102400 --memory-cell-bytes 1
```

`train_fast.py` needs the upstream `tb_baselines` with the `mini_trunk` architecture patch.

## Evidence

- `v9/gate_v11w.log`: 96 games a map group against the 64-channel model, the teacher and a TigerAnts clone.
- `v9/GAME_NOTES.md`: the rules, the 76 boards and the field. `v9/HANDOFF.md`: the full working log.

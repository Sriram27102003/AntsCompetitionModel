# FatAss — `v11m_mini`

The mini-class entry: `models/v11m_mini/` (`model.onnx` + `manifest.json`), 1,021,459 bytes, weight class **mini**.
On qa.tinybrains.dev (Hive Minds) it played as **FatAss v7**: 291 wins in 329 games, one strike.

## What it is

The 96-channel model (`v11h_small`, on the `v11h-small` branch) is 15 KB over the mini cap. This is the
same network with its two least-used channels dropped at every layer (96 -> 94 channels, 494,997
parameters), with **no retraining**: on 38,384 test decisions it picks the same move 99.8% of the time
and agrees with the teacher exactly as often (88.1%). Weights: `v9/runs/v11m/init.pt`.

It carries the same in-graph rules (`v9/safe_policy.py`): own-ant collisions removed in 8 passes, and
an undefended enemy hill beside an ant is taken.

## Rebuild it

```sh
cd v9
export PYTHONPATH=<path to Tiny-Brains/ants>/baselines/src:$PWD TINYBRAINS=<path to tinybrains CLI>
python trim.py runs/v11w/best.pt 94 runs/v11m/init.pt          # needs the 96-channel checkpoint
python safe_policy.py $PWD/runs/v11m/init.pt $PWD/../models/v11m_mini --take-hills --rounds 8
tinybrains check ../models/v11m_mini/model.onnx ../models/v11m_mini/manifest.json --memory-flat-bytes 102400 --memory-cell-bytes 1
```

`v9/GAME_NOTES.md` has the rules, boards and field; `v9/HANDOFF.md` is the full working log.

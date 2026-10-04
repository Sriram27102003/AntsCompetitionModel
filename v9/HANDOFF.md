# v9 workspace handoff (written 2026-10-01 23:52 IST)

## Goal
Top the **release** leaderboard at tinybrains.dev. qa.tinybrains.dev (season "Hive Minds") is only a
testbed; rivals can download our QA replays, so ask before putting the best model there.
As of this note production has no open season (only the closed Nuptial Flight).

## Status (updated 2026-10-02 17:20 IST) — read this first

**On the QA ladder now (both submitted at the user's request):**
| Model | Version | Files | Class | Notes |
|---|---|---|---|---|
| FatAss II (3d3f0c15-…) | v1 (44e0ebb7-…) | `../models/v11h_small` | small | #1 on Open at 40.16 after 63 games: 61 wins, MadPanda 53-0-2, no strikes; worst turn 495 ms on a 96x96 board |
| FatAss (6934f5eb-…) | v7 (4f410469-…) | `../models/v11m_mini` | mini | replaced v5 (v6) at 17:13 IST by the user's choice; 1,021,459 bytes |

`v11m_mini` is `runs/v11m/init.pt`: v11w with its two least-used channels per layer dropped (`trim.py`),
no retraining; it picks the same move as v11w 99.8% of the time. 96 channels is 15 KB over the mini cap
and the weights alone leave only ~15 KB for rules that need ~29 KB, so the exact model cannot be mini;
micro (128 KB) would need a network about a tenth the size.

**Running:** `round3_mini.sh` (log `round3_mini.log`): waits for the dagger3 collectors, packs, trains
`runs/v11n` (94 channels, from `runs/v11m/init.pt`, 2 epochs on every shard), gates against v11w, the
teacher and the MadPanda clone -> `gate_v11n.log`. Ends with `ROUND3_MINI_DONE`, around midnight IST.
Submit nothing further unasked.

## Earlier status (updated 2026-10-02 16:45 IST)

**Submitted, at the user's request:** `../models/v11h_small` went to QA as a second model, **FatAss II**
(model_id 3d3f0c15-3756-4950-9028-57731b0039bc), version 1, version_id 44e0ebb7-77f3-41de-9fc9-c42b36975946.
FatAss (v5 playing as v6) keeps its own slot. How: in the signed-in built-in browser, same-origin
`POST /v1/submissions` with both sha256 hashes, then `curl -T` each file to the one-shot URLs.
Follow it at `GET /v1/me/versions/<version_id>` (signed in) or `/v1/versions/<version_id>` once active.
Round 3 (`dagger3_v11.sh`) was stopped at the user's request; its partial data is `data/dagger3-*.partial`.
**Next:** when FatAss II has played, re-run `fetch_replays.py`, read its record against MadPanda, FireAnts
and TigerAnts and its strikes by board size, then decide what to train. Submit nothing further unasked.

## Earlier status (updated 2026-10-02 15:45 IST)

**Do not submit anything without the user's go-ahead.** Two candidates are ready; the user chooses.

| Folder | Weights | In-graph rules | Size | Local probe (v5 = 43 ms) |
|---|---|---|---|---|
| `../models/v10h_mini` | v10e, 64 channels | collisions 8 passes + take undefended hill | 494,417 B (mini) | 64 ms, 1.49x v5 |
| `../models/v11h_small` | v11w, 96 channels | same | 1,063,636 B (small) | 89 ms, 2.08x v5 |

Older exports (`v10s`, `v10e`, `v10f`, `v10g`, `v11w_small`) are kept; `v10s/e/f/g` are equal in strength
(four-way CLI test on all 30 two-player boards: 28-2-30 between the two weight sets).

**The wide net (`runs/v11w`, 87.8% held-out agreement, trained 2 epochs from v10e with `--channels 96 --batch 32`)**
- Gate (`gate_v11w.log`, 96 games a group): vs v10e 7-0-9 (2p), 61-7-28 (4p), 89-33-97 (6p);
  vs the teacher 8-0-7, 43-3-50, 63-28-101 (v10e: 1-0-13, 31-3-62, 80-19-117).
  Points a game: 2.56 / 4.46 / 4.83 against v10e's 1.91 / 2.44 / 4.32.
- Real CLI (`arena/v11w-*`): vs `v10s_mini` 20-4-8 (2p), 32-7-21 (4p); vs `v10f_mini` 14-4-14, 33-10-17. No strikes locally.

**Platform speed is a real constraint (new).** On QA, v5 averages 93 ms a turn on 96x96 boards with
per-game worst turns up to 465 ms, and has taken 8 strikes, all on boards of 6,144+ cells (a quarter of
its games). The platform is about 4x slower than this Mac's `tinybrains check` probe. A strike holds every
ant for the turn; 5 in a game forfeit. Scaling v5's per-game worst turns, the slowest turn would pass
500 ms in about 5% of big-board games at 1.46x v5, 13% at 1.84x, 25% at 2.42x (an undercount).
Each collision pass costs 1.2-2 ms locally; 8 passes leave 0.025% of ants colliding, 16 leave none.
A convolution rewrite of the passes gives identical moves but ran slower (`Safe.slow = False`), so the rolled passes are exported.

**QA field (1,052 matches fetched, to 2026-10-02 07:06 UTC):** MadPanda (christopheryang, micro) is #1 at
33.58; FireAnts 28.53, HiveMicro 28.32, TigerAnts 27.90, FatAss v5 27.80. MadPanda vs v5: 89-5-24.
It almost never holds (0.2%) and reverses direction on half its moves. `runs/clone-MadPanda` (79.1%
agreement) beats the TigerAnts clone 17-2-0 like the original does, and loses to the wide net 0-0-22 and
to v10e 2-0-18 on 2p (`gate_madpanda_2p.log`). Clones are weaker than originals: encouraging, not proof.
The replay conversion is verified: the v5 checkpoint reproduces 100% of FatAss's recorded ladder moves.

**MPS memory:** training needs `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6`
and `train_fast.py` now empties the MPS cache every 50 steps; without that the 96-channel run reached 19 GB.

## Earlier status (updated 2026-10-02 09:15 IST)

The user is asleep and asked for the best model to be prepared without them. **Do not submit anything.**

**Running:** `train_fast.py packed/*.npz --init runs/v10e/best.pt --channels 96 --epochs 2 --lr 5e-4
--fight-weight 2 --out runs/v11w` (started 08:58, log `train_v11w.log`, about 2.3 h an epoch, ends ~13:35).
The 64-channel net underfits (train 85.1% = held-out 85.3%), so this widens it to 96 channels; the
warm start reproduces v10e exactly (checked: max difference 0.0). At 516k parameters it exports as class
`small` (`safe_policy.py ... --small`). Run nothing heavy beside it: swap is at 3 GB, free disk 2 GB.

**Round 2 result (`gate_v10e.log`):** v10e vs v10d, both collision-free: 2p 8-0-1, 4p 39-1-20, 6p 55-12-67.
v10e vs the strong teacher: 2p 2-0-5, 4p 28-4-28, 6p 40-11-69.

**Candidates on disk (nothing submitted):**
| Folder | Weights | In-graph rules | Size, time |
|---|---|---|---|
| `../models/v10s_mini` | v10d | collisions, 8 passes | 500,057 B, 92 ms |
| `../models/v10e_mini` | v10e | collisions, 8 passes | 500,057 B, 54 ms |
| `../models/v10f_mini` | v10e | collisions, 16 passes + take an undefended adjacent enemy hill | 534,594 B, ~90 ms |

**Findings since 04:20**
- Scoring check (`duel.py` now prints points: razed, lost, first raze turn, endings). v10e stepped onto
  an undefended enemy hill beside it only 79% of the time; `safe_policy.py --take-hills` makes that a rule.
- 8 resolution passes left a few collisions in colonies of 25+; 16 leaves none (102k ants checked).
- Two copies of one policy tie (195 of 200): the game is symmetric. Judge a small change against a
  different opponent, never in a mirror.
- Teacher settings (`tune_teacher.log`, 160 two-player games each against `strong:take_hill=1`): loss_behind
  1.0 and 1.3 worse (61-94, and 1.64 vs 2.06 points), 2.2 and loss_ahead 0.7 level, guard_radius2 60 much
  worse, 12 level. `take_hill=1` level with the old teacher (78-9-73). The settings are at a local optimum.
- Knowing every enemy hill from turn 0 (`strong:know_hills=1`, a lookup by board size and own hill; 29 of 30
  two-player boards put the enemy hill at own hill + half the board) did not make this teacher stronger.
- Production: Anxiety takes 93% of the available points on 2p boards in a median 69 turns and has a
  median 29 ants at turn 40. Its opponents were weaker than ours are in local duels, so not like for like.
- The starter kit on the QA site is `starter_new/`; `v10s_mini` beat its three models in all 88 CLI games.
- `duel.py` policy specs: `ckpt:`, `safe:` (collision-free), `hill:` (+ take hills), `route:` (+ a blocked ant
  takes its next best move; built, not yet judged against a different opponent).

**Next:** gate `safe:`/`hill:runs/v11w/best.pt` against `hill:runs/v10e/best.pt`, the teacher and the
TigerAnts clone; export the winner with `safe_policy.py ... --take-hills [--small]`; confirm it in `arena.py`.

## Earlier status (updated 2026-10-02 04:20 IST)

**Running:** `./dagger2_v10.sh` (started 04:14 under `nohup caffeinate -i`). Steps: collect `dagger2-self` (60k rows)
and `dagger2-mixed` (50k) with the student playing collision-free; relabel the production champions'
positions (`strong-champ`); pack; train `runs/v10e` from `runs/v10d/best.pt` on every shard, old ones
included, `--fight-weight 2`; gate -> `gate_v10e.log`. Progress: `grep '^\[' dagger2_v10.log`; ends with
`DAGGER2_DONE`, roughly 4 hours. **The external drive must stay plugged in**: `v9/data`, `v9/packed`
and `../data` are now links into `/Volumes/Thaniska/AntsCompitionModel/` (git shows `../data` as deleted).

**New since the 02:12 note**
- `GAME_NOTES.md`: rules from the engine source, all 76 boards, both fields. Rank is score only
  (+2 raze, -1 hill lost); QA games last a median 126 turns; ladder games run to 1000 turns, not 500.
- **Own-ant collisions were the biggest leak** (both ants die; v10d collided on 6.4% of turns).
  `safe_policy.py` removes them inside the exported graph, no retraining. `../models/v10s_mini/` =
  v10d weights + that fix: 500,057 bytes, 92 ms, passes `tinybrains check`. In `duel.py` it is `safe:<ckpt>`.
  - 2p (1000 turns): v10d+safe vs v10d 20-0-1; vs strong teacher 6-0-8.
  - 4p: vs v10d 47-7-6; vs strong teacher 27-2-31.
  - Real CLI, exported files (`arena/v10s-2p`): v10s_mini beat v10_mini 26-0 and v5 26-0, no strikes.
- `replays_prod/`: all 1,279 production 2/4/6p games with a top-five model; they re-simulate exactly on
  engine 819166. `data/champ-anxiety` (47.7k rows) and `data/champ-riot` (31.2k) are those models' moves.
  `champion_study.py` compares them with the teacher (same move 70% / 66% of the time).
- `runs/clone-Anxiety` (stopped at epoch 3, 79.7%) is weak in play (loses to the teacher 0-17 on 2p):
  a clone does not carry the champion's strength, so it is sparring only, not a yardstick.
- `train_fast.py --fight-weight`, `collect_strong.py dagger --safe` are new options.
- Machine: run one heavy job at a time. Training plus a duel pushed swap to 4 GB and free disk to 143 MB.

**When the round finishes:** read `gate_v10e.log` (`v10e+safe vs v10d+safe`, and each vs `strong`).
If v10e is not worse anywhere, export with `python safe_policy.py $PWD/runs/v10e/best.pt <abs path>/models/v10e_mini`
(absolute paths; keep `v10s_mini`). Nothing has been submitted; do not submit without the user's go-ahead.

## Earlier status (updated 2026-10-02 02:12 IST)
Nothing is running. DAgger round 1 finished and **v10 was updated**:

- `../models/v10_mini/` = the DAgger-round-1 model (weights `runs/v10d/best.pt`, 85.2% held-out
  agreement with the strong teacher). Passes the Hive Minds check: 467,803 bytes, 39 ms, memory OK.
- `../models/v10_mini_round0/` = the previous v10 (weights `runs/v10/epoch1_backup.pt`), kept as a backup.
- Gate (`gate_v10d.log`, plus a 300-game 2p run): new vs old v10 = 19-6-5 (2p), 21-4-11 (4p),
  37-17-31 (6p). New v10 vs v5: 30-2-0, 22-5-8, 46-16-23. vs TigerAnts clone: 29-1-0, 20-12-4, 46-24-15.
- The strong teacher still beats it about 80-1 on 6p and 35-1 on 4p: one DAgger round was a modest
  gain, not a change of level.

## Next options
- Another DAgger round: copy `dagger_v10.sh`, set `S=runs/v10d/best.pt`, new seeds and output names.
  Expect diminishing returns; the model may be at what MiniTrunk(64,6) can imitate.
- A bigger or deeper network (size no longer earns a separate ladder; the 500 ms turn is the limit,
  and v10 uses 39 ms), or more training epochs.
- Nothing has been submitted except v5, re-entered on QA as FatAss version 6 (#3, 27.83).
  Do not submit without the user's go-ahead; check tinybrains.dev for a release season first.

## Results so far
- Strong teacher (`strong_teacher.py`) beats v5, the upstream teacher and both clones almost every game.
- v10 = MiniTrunk(64,6) + 2 memory planes, behaviour-cloned from the strong teacher (epoch 1,
  83.1% held-out agreement). Passes the Hive Minds admission check (467,803 bytes, 40 ms).
  vs v5: 6-0 (2p), 26-1-9 (4p), 40-18-27 (6p). vs TigerAnts clone: 6-0-2, 18-2-14, 47-12-26.
- FatAss's ladder weakness: colony stops growing (~8 ants) once contact starts around turn 20.

## Environment
    export PYTHONPATH=/Users/sriram/Documents/tinybrains-ants/baselines/src
    export TINYBRAINS=$PWD/bin/tinybrains        # CLI 0.4.0; brew still has 0.3.0 (old engine)
    PY=../.venv/bin/python
`games.toml` here and in `../../tinybrains-ants/baselines/` pins engine 819166… (Hive Minds).
The upstream clone is patched: `nets.py` has `mini_trunk`; `train/bc.py` has `--init`.

## Files
| File | Purpose |
|---|---|
| `strong_teacher.py` | the teacher (food assignment, focus-rule fight simulation, guards, field for spare ants) |
| `duel.py` | head-to-head harness: `strong`, `teacher`, `ckpt:<path>`; reports ahead/tied/behind and ants at turns 25/50 |
| `collect_strong.py` | `rollout` (teacher self-play), `relabel` (ladder positions), `dagger` (student plays, teacher labels) |
| `prep_fast.py` / `train_fast.py` | bit-pack a dataset once / train from the packed shards |
| `fetch_replays.py` / `replays_to_dataset.py` | download every QA replay / re-simulate into training rows |
| `arena.py` | round robin of exported ONNX models through the real CLI |
| `runs/clone-TigerAnts`, `runs/clone-FireAnts` | sparring partners cloned from replays (77.5% / 67% agreement) |
| `maps_hive/` | the 76 Hive Minds boards |

## Machine limits
16 GB RAM, disk ~98% full (about 5 GB free). Run one training job at a time; two pushed the Mac
into swap and filled the disk. `packed/` is 1.85 GB.

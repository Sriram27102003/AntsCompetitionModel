# TinyBrains Ants: rules, boards and the field (compiled 2026-10-02)

Sources: engine source (`../../tinybrains-ants/engine/src`, engine 819166…), the season definitions at
`https://qa.tinybrains.dev/v1/games` and `https://tinybrains.dev/v1/games`, the 76 boards in `maps_hive/`,
450 QA matches (`replays_qa/`) and all 10,182 production matches (`prod_matches.json.gz`).

## Rules that decide games

| Rule | Value | Consequence |
|---|---|---|
| Turn order | move, attack, raze, spawn ants, gather food, spawn food | food gathered this turn becomes an ant next turn at the earliest |
| Score | +1 per own hill at the start, +2 per enemy hill razed, -1 per own hill lost | **rank is score only**; ants and food count for nothing by themselves |
| Collision | every ant ending on a shared square dies, own ants included | a policy that sends two ants to one square loses both |
| Blocked move | an order into water or into food is ignored, the ant stays | approach food from beside it |
| Attack radius² | 5 | focus rule: an ant dies if any enemy in range has focus <= its own; deaths do not cascade |
| Gather radius² | 1 | an ant must stand next to food; food in reach of two players is destroyed |
| View radius² | 77 (about 8.8 cells) | foes, food and hills are visible only this turn; **known water is remembered by the engine** |
| Spawn | one ant per food in the hive, on a free own hill, least recently touched first | an ant standing on its own hill blocks that hill |
| Raze | an enemy ant alive on a hill after the fight razes it | an ant that dies on a hill razes nothing |
| Food | hidden rate per match: 5-11 food x players every 19-37 turns, in symmetric sets | every seat is fed equally; no square can be camped |
| Endings | lone survivor (takes all standing hills), rank stabilised, one side holds 85% of population for 150 turns, turn limit | games end as soon as nobody can change the order |
| Symmetry | every board is the same board for every seat, moved by a fixed shift (`symmetry.dr/dc`) | enemy hills sit at own hill + k x shift |

Ladder execution (Hive Minds): 1000 turns max, 500 ms a turn, 5 strikes. Local gates so far ran 500 turns.

## Season structure

| | QA: Hive Minds | Production: Nuptial Flight |
|---|---|---|
| State | open until 2026-12-29; round 1 starts 2026-10-07, 100 games a round, then finals | closed 2026-09-30; no open season |
| Engine | 819166… | cd656… (same game rules: 12 of 12 production replays re-simulate to the same ending on 819166) |
| Ladder | one Open rating; classes nano 16 KB to large 64 MB all play each other | Open plus per-class ladders |
| Memory | 100 KB flat + 1 byte per cell | none |
| Boards | 76 boards, 2/4/6 players, sides 24-96 | 32 boards, 2-8 players |
| Field | 3 rivals' models + 3 baselines | 69 models |
| Entries | 3 per user | |

## The 76 Hive Minds boards (`maps_hive_stats.json`)

| Players | Boards | Cells | Water | Nearest enemy hill (walking steps) | Hills per seat |
|---|---|---|---|---|---|
| 2 | 30 | 576-9216 | 22-56% | 24-78, median 30 | 1 on 23 boards, 2 on 5, 3 on 2 |
| 4 | 34 | 1296-9216 | 23-56% | 13-48, median 22 | 1 on 12, 2 on 17, 3 on 5 |
| 6 | 12 | 2304-9216 | 22-54% | 14-48, median 23 | 1 on 2, 2 on 7, 3 on 3 |

Families: maze 27, cave 26, rooms 21, open 2. Nine sizes from 24x24 to 96x96.
QA has so far played only 2-player (286) and 4-player (164) matches.

## How ladder games are actually won

- QA: median game is 126 turns. 277 of 450 end "rank stabilised", 112 "lone survivor", 52 at the turn limit.
- QA 2-player: 209 of 286 end 3-0, i.e. the winner razed the loser's only hill and kept its own.
- Production medians: 135 turns (2p), 185 (4p), 565 (6p), 934 (8p).
- So the contest is: reach and raze enemy hills first while holding your own. Economy only matters as the way to get there.

## The field

QA (mean rank, share of games won outright):
TigerAnts (micro) 1.24 / 75%; FireAnts (nano) 1.43 / 73%; FatAss v5 (mini) 1.70 / 42%;
baselines Carpenter 24%, Carebara 18%, Redwood 15%.
FatAss v5 head to head (ahead/tied/behind): FireAnts 18/2/43, TigerAnts 2/3/7.

Production, top of 69 (share of games won outright):
Anxiety (large) 96%; Riot Control (small) 92%; GOD of DEATH (Ryuk) (large) 88%; Perfectionism (micro) 84%;
Gaemi (nano) 69%; Dr. More Papers (nano) 66% — all six by Al-Scripting. Then 4k, 6k (Randomblock1), gambino and
Halyax (paulgowdy77), riot (kevinCCme), RAZE.
FatAss v5 was 44th of 69 by mean finishing position, and finished ahead of Perfectionism in 4 of 93 meetings.
Eight of the top fifteen are nano (16 KB) models: policy quality, not size, separates the field.

## Data on disk

- `replays_qa/`: all 450 QA matches. `replays_prod/`: every 2/4/6-player production match with a top-five model.
- `replays_to_dataset.py <dir> --models NAME… --out data/x.jsonl.gz` turns either into training rows.

## The starter kit the QA site points to (checked 2026-10-02)

`https://qa.tinybrains.dev/start` links to `github.com/Tiny-Brains/ants-starter`; its last commit is
2026-09-29 (`a9610c6`), cloned here as `starter_new/`. The copy in `../ants-starter/` is older and pins the
closed season's engine.

| What changed in the kit | Where we stand |
|---|---|
| Pins engine 819166 (Hive Minds) | v9 already plays on 819166 |
| New memory baseline `models/nano-bc-max-r` and `train.py --memory` (two planes kept with Max) | v10 already uses the same two memory planes; the model is in `opp/nano-bc-max-r` |
| Training code is `Tiny-Brains/ants/baselines` | `../../tinybrains-ants` is at upstream's latest commit (`54edfab`), plus our MiniTrunk and `--init` patches |

The kit's recipe is one scripted teacher cloned into a 3,006-parameter net; our strong teacher and v10
are past it. Its four models (nano-bc, nano-bc-max-r, micro-bc, micro-percell) are useful as fixed
opponents: the ladder's baselines hold most of the seats on QA (853 of 1,228), so a candidate must
keep beating them while it improves against the leaders.
The page also says a model "gets no score and no turn number", and that classes run from nano
(16 KiB) to large (64 MiB), each with 100 KiB + 1 byte a cell of memory.

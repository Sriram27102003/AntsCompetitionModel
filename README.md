# AntsCompetitionModel

AI policy models, training pipelines, datasets, and tournament evaluation suites for the **TinyBrains Ants Competition**.

## Overview

This repository contains the training code, datasets, evaluation suites, and exported ONNX models for competitive Ants game matches. The active championship policy is a **MiniTrunk (64, 6)** architecture in the **mini** weight class, trained with tactical teacher distillation and augmented season data.

### Championship Model Specifications (`v5_mini`)

| Metric / Parameter | Value |
|---|---|
| **Weight Class** | **mini** — 468,384 bytes (~45% of 1 MB cap) |
| **Parameters** | 230,255 (fp16 initializers) |
| **Architecture** | `MiniTrunk`, receptive field: 15 cells each way |
| **Dilations** | `[1, 2, 4, 4, 2, 1, 1]` |
| **Validation Accuracy** | 95.2% (Validation loss: 0.1611) |
| **Worst-case Inference Time** | 43.25 ms (~4.3% of 1000 ms turn budget) |
| **ONNX Operators** | `Cast`, `Concat`, `Constant`, `Conv`, `Relu`, `Slice` |

---

## Directory Structure

```
.
├── model.onnx              # Primary championship ONNX policy
├── manifest.json           # Model input/output tensor declarations & adapters
├── metrics.json            # Model validation & benchmark metrics
├── card.md                 # Detailed model specification card
├── classes.toml            # Competition weight class limits & rules
├── games.toml              # Ants cartridge pin & engine verification
│
├── src/                    # Custom architectures, feature extractors, trainers
├── data/                   # Tactical teacher and augmented season datasets (.jsonl.gz)
├── models/                 # Baseline opponent checkpoints and candidate models
├── matches/                # Match configuration definitions
├── replays/                # Self-play and evaluation match replays
├── season_maps/            # Competition tournament map configurations
│
├── benchmark_suite.json    # Benchmark match configuration
├── suite_*.json            # Head-to-head, 3-player, and 4-player FFA tournament suites
├── ants-starter/           # Starter baseline toolkit and reference models
└── build_championship_bot.py # Script to package and export competition bots
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- [TinyBrains CLI](https://github.com/Tiny-Brains/cli):
  ```sh
  brew tap tiny-brains/cli https://github.com/Tiny-Brains/cli
  brew install tiny-brains/cli/tinybrains
  ```

### Verify Submission Model

Verify model admission against the tournament engine rules:

```sh
tinybrains check model.onnx manifest.json
```

### Run Matches & Evaluation

Run a match against baseline bots:

```sh
tinybrains matches/self-play.json
tinybrains view replays/self-play.json
```

Run tournament suites:

```sh
tinybrains benchmark_suite.json
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

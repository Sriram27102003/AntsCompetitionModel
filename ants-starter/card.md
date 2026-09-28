# ants-starter

A micro-class Ants policy.

| | |
|---|---|
| Weight class | **micro** — 54,430 of 131,072 bytes (42% of the cap) |
| Parameters | 24,077 (fp16 initializers) |
| Architecture | `Trunk`, receptive field **15 cells** each way (dilations [1, 2, 4, 8]) |
| Method | behaviour cloning, 3 epochs over 10,000 seat-turns, 84.8% held-out agreement, seed 7 |
| Adapter | 208,423 of 1,000,000 operations at its worst reference case |
| Inference | 26.02 ms at the worst reference case — **2.6%** of the 1000 ms a seat gets |
| Operators | Cast, Concat, Constant, Conv, Relu, Slice |
| Engine | `sha256:cd656bc84c1accec25106c035dece8cdac211e1e0f1a5aa6a2e7836c1cfc3bb4` |
| Model hash | `sha256:59c1c87340d216df0fc50980a22fbe489f1961fafe3aff7f5581bd7df539d2ab` |
| Manifest hash | `sha256:0f8f8ed7b8fe317d53cc072ce75aa835ec2fdeb21dea8dd6528b6fd77a985234` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap. It is here because the turn deadline is what a graph too expensive to
play runs into, and a seat's share of it is the WHOLE turn -- one `model_infer` call per seat,
each with its own deadline.

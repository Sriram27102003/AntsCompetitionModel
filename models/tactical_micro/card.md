# tactical_micro

A micro-class Ants policy.

| | |
|---|---|
| Weight class | **micro** — 54,432 of 131,072 bytes (42% of the cap) |
| Parameters | 24,077 (fp16 initializers) |
| Architecture | `Trunk`, receptive field **15 cells** each way (dilations [1, 2, 4, 8]) |
| Method | Tactical BC: 6 epochs, 75,000 seat-turns, 80.1% val_acc, seed 42 |
| Adapter | 208,423 of 1,000,000 operations at its worst reference case |
| Inference | 16.92 ms at the worst reference case — **1.7%** of the 1000 ms a seat gets |
| Operators | Cast, Concat, Constant, Conv, Relu, Slice |
| Engine | `sha256:cd656bc84c1accec25106c035dece8cdac211e1e0f1a5aa6a2e7836c1cfc3bb4` |
| Model hash | `sha256:43f98f82b7808f61995f1cd4261ac879f4a3962840b8b57ccda5040e78c1a590` |
| Manifest hash | `sha256:2d290885dc65f1d90578250d2a7590b9a8af366004bba2ee9fc5eef7a98e9fba` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap. It is here because the turn deadline is what a graph too expensive to
play runs into, and a seat's share of it is the WHOLE turn -- one `model_infer` call per seat,
each with its own deadline.

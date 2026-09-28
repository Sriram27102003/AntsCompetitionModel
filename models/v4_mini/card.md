# v4_mini

A mini-class Ants policy.

| | |
|---|---|
| Weight class | **mini** — 319,574 of 1,048,576 bytes (30% of the cap) |
| Parameters | 156,379 (fp16 initializers) |
| Architecture | `Trunk`, receptive field **23 cells** each way (dilations [1, 2, 4, 8, 8]) |
| Method | FatAss v4 Championship: Trunk(64,4), 3 epochs, 84.0% val_acc, 0 strikes |
| Adapter | 208,423 of 1,000,000 operations at its worst reference case |
| Inference | 37.64 ms at the worst reference case — **3.8%** of the 1000 ms a seat gets |
| Operators | Cast, Concat, Constant, Conv, Relu, Slice |
| Engine | `sha256:cd656bc84c1accec25106c035dece8cdac211e1e0f1a5aa6a2e7836c1cfc3bb4` |
| Model hash | `sha256:3143602d01a3a067affc0073b2c104d0741ac0eb96be2e97da41a2e42fd83c32` |
| Manifest hash | `sha256:1ba73e2c31519aa7beb32cf0ddbe02fd79ce05c4079b79045441656169a2d39a` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap. It is here because the turn deadline is what a graph too expensive to
play runs into, and a seat's share of it is the WHOLE turn -- one `model_infer` call per seat,
each with its own deadline.

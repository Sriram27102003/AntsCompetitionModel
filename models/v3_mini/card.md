# v3_mini

A mini-class Ants policy.

| | |
|---|---|
| Weight class | **mini** — 184,586 of 1,048,576 bytes (18% of the cap) |
| Parameters | 88,891 (fp16 initializers) |
| Architecture | `Trunk`, receptive field **23 cells** each way (dilations [1, 2, 4, 8, 8]) |
| Method | FatAss v3: Trunk(48,4), 3 epochs over 120k season turns, 79.6% val_acc, 0 strikes |
| Adapter | 208,423 of 1,000,000 operations at its worst reference case |
| Inference | 32.24 ms at the worst reference case — **3.2%** of the 1000 ms a seat gets |
| Operators | Cast, Concat, Constant, Conv, Relu, Slice |
| Engine | `sha256:cd656bc84c1accec25106c035dece8cdac211e1e0f1a5aa6a2e7836c1cfc3bb4` |
| Model hash | `sha256:01b26151737a4cce7bd2f7ed9d73b56136e8b0d0ed6a7ba0458f0042a39bd415` |
| Manifest hash | `sha256:b641479edfd4ec087ab4c3d3e03a1120d08c944e2789850395b861e5fb0a60d3` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap. It is here because the turn deadline is what a graph too expensive to
play runs into, and a seat's share of it is the WHOLE turn -- one `model_infer` call per seat,
each with its own deadline.

# micro-bc-20ep

A micro-class Ants policy.

| | |
|---|---|
| Weight class | **micro** — 54,431 of 131,072 bytes (42% of the cap) |
| Parameters | 24,077 (fp16 initializers) |
| Architecture | `Trunk`, receptive field **15 cells** each way (dilations [1, 2, 4, 8]) |
| Method | untrained |
| Adapter | 208,423 of 1,000,000 operations at its worst reference case |
| Inference | 16.64 ms at the worst reference case — **1.7%** of the 1000 ms a seat gets |
| Operators | Cast, Concat, Constant, Conv, Relu, Slice |
| Engine | `sha256:cd656bc84c1accec25106c035dece8cdac211e1e0f1a5aa6a2e7836c1cfc3bb4` |
| Model hash | `sha256:3aa11eb968855c6653a1e20194cad5c3822515d9e4c04c8cf067fbac6d9fa716` |
| Manifest hash | `sha256:abb1342bbd06b3a08b74c00cdc5fff800c9c28bfe65d0aa5a70936397216042a` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap. It is here because the turn deadline is what a graph too expensive to
play runs into, and a seat's share of it is the WHOLE turn -- one `model_infer` call per seat,
each with its own deadline.

# test_mini

A mini-class Ants policy.

| | |
|---|---|
| Weight class | **mini** — 413,210 of 1,048,576 bytes (39% of the cap) |
| Parameters | 202,671 (fp16 initializers) |
| Architecture | `Trunk`, receptive field **39 cells** each way (dilations [1, 2, 4, 8, 8, 8, 8]) |
| Method | test mini |
| Adapter | 0 of 1,000,000 operations at its worst reference case |
| Inference | 0.00 ms at the worst reference case — **0.0%** of the 1000 ms a seat gets |
| Operators | Cast, Concat, Constant, Conv, Relu, Slice |
| Engine | `sha256:cd656bc84c1accec25106c035dece8cdac211e1e0f1a5aa6a2e7836c1cfc3bb4` |
| Model hash | `sha256:772acbeca180ad752f10955e8968fbe7046fe362b3ed0856918fb7c6a1c3858d` |
| Manifest hash | `sha256:8de5340c90ba6d5ab02c0cbe3a2a60ddd9e7227ea3ded2778a0bd8e30d0375c2` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap. It is here because the turn deadline is what a graph too expensive to
play runs into, and a seat's share of it is the WHOLE turn -- one `model_infer` call per seat,
each with its own deadline.

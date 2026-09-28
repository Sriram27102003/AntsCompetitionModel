# test_export

A mini-class Ants policy.

| | |
|---|---|
| Weight class | **mini** — 468,388 of 1,048,576 bytes (45% of the cap) |
| Parameters | 230,255 (fp16 initializers) |
| Architecture | `MiniTrunk`, receptive field **15 cells** each way (dilations [1, 2, 4, 4, 2, 1, 1]) |
| Method | test |
| Adapter | 208,423 of 1,000,000 operations at its worst reference case |
| Inference | 39.35 ms at the worst reference case — **3.9%** of the 1000 ms a seat gets |
| Operators | Cast, Concat, Constant, Conv, Relu, Slice |
| Engine | `sha256:cd656bc84c1accec25106c035dece8cdac211e1e0f1a5aa6a2e7836c1cfc3bb4` |
| Model hash | `sha256:3c703b68d10987206bb2f89f67d85f0e2ad481cf0493eed1f153f31959325c90` |
| Manifest hash | `sha256:a60c1cc7693b04e056d2aef825120dfe1fa2f8f6668a2e26e4a7adb94fb2f6ce` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap. It is here because the turn deadline is what a graph too expensive to
play runs into, and a seat's share of it is the WHOLE turn -- one `model_infer` call per seat,
each with its own deadline.

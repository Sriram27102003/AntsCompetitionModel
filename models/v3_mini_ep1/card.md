# v3_mini_ep1

A mini-class Ants policy.

| | |
|---|---|
| Weight class | **mini** — 184,590 of 1,048,576 bytes (18% of the cap) |
| Parameters | 88,891 (fp16 initializers) |
| Architecture | `Trunk`, receptive field **23 cells** each way (dilations [1, 2, 4, 8, 8]) |
| Method | Championship v3 ep1: Trunk(48,4), 72.2% val_acc, 0 strikes |
| Adapter | 208,423 of 1,000,000 operations at its worst reference case |
| Inference | 28.90 ms at the worst reference case — **2.9%** of the 1000 ms a seat gets |
| Operators | Cast, Concat, Constant, Conv, Relu, Slice |
| Engine | `sha256:cd656bc84c1accec25106c035dece8cdac211e1e0f1a5aa6a2e7836c1cfc3bb4` |
| Model hash | `sha256:210073632ffd7158dffde5d995ac0f7e677b66835171b9125b9e75da3c74863e` |
| Manifest hash | `sha256:1c4b875effeac0fa19a02493a56781566adc05ca73b628753e83661d83b0ba8a` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap. It is here because the turn deadline is what a graph too expensive to
play runs into, and a seat's share of it is the WHOLE turn -- one `model_infer` call per seat,
each with its own deadline.

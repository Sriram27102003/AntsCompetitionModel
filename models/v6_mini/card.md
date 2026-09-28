# v6_mini

A mini-class Ants policy.

| | |
|---|---|
| Weight class | **mini** — 468,384 of 1,048,576 bytes (45% of the cap) |
| Parameters | 230,255 (fp16 initializers) |
| Architecture | `MiniTrunk`, receptive field **15 cells** each way (dilations [1, 2, 4, 4, 2, 1, 1]) |
| Method | FatAss v6 Fused: MiniTrunk(64,6), 5 epochs, 88.5% val_acc, val_loss 0.3263 |
| Adapter | 208,423 of 1,000,000 operations at its worst reference case |
| Inference | 48.82 ms at the worst reference case — **4.9%** of the 1000 ms a seat gets |
| Operators | Cast, Concat, Constant, Conv, Relu, Slice |
| Engine | `sha256:cd656bc84c1accec25106c035dece8cdac211e1e0f1a5aa6a2e7836c1cfc3bb4` |
| Model hash | `sha256:5d28cb9eec7505d265c94631c346ba1f96659b82da7516f9a7009d889eb69d02` |
| Manifest hash | `sha256:386b842387ebbd760268812e9233a009a39c62cdc66127c5da7ce452b1019dcf` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap. It is here because the turn deadline is what a graph too expensive to
play runs into, and a seat's share of it is the WHOLE turn -- one `model_infer` call per seat,
each with its own deadline.

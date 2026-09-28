# v5_mini

A mini-class Ants policy.

| | |
|---|---|
| Weight class | **mini** — 468,384 of 1,048,576 bytes (45% of the cap) |
| Parameters | 230,255 (fp16 initializers) |
| Architecture | `MiniTrunk`, receptive field **15 cells** each way (dilations [1, 2, 4, 4, 2, 1, 1]) |
| Method | FatAss v5 Championship: MiniTrunk(64,6), 15 epochs, 95.2% val_acc, val_loss 0.1611 |
| Adapter | 208,423 of 1,000,000 operations at its worst reference case |
| Inference | 43.25 ms at the worst reference case — **4.3%** of the 1000 ms a seat gets |
| Operators | Cast, Concat, Constant, Conv, Relu, Slice |
| Engine | `sha256:cd656bc84c1accec25106c035dece8cdac211e1e0f1a5aa6a2e7836c1cfc3bb4` |
| Model hash | `sha256:fff37c8adc25807df7649e952b8494693efb313379d4efdff963a3a7b079155c` |
| Manifest hash | `sha256:54be87011aab1e3c8c26fbce718f9fca9231171fa3d7dd9b999e975bc21146d2` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap. It is here because the turn deadline is what a graph too expensive to
play runs into, and a seat's share of it is the WHOLE turn -- one `model_infer` call per seat,
each with its own deadline.

# championship_mini

A mini-class Ants policy.

| | |
|---|---|
| Weight class | **mini** — 468,394 of 1,048,576 bytes (45% of the cap) |
| Parameters | 230,255 (fp16 initializers) |
| Architecture | `MiniTrunk`, receptive field **15 cells** each way (dilations [1, 2, 4, 4, 2, 1, 1]) |
| Method | Championship Mini-Trunk: 10 epochs, 120,000 seat-turns across 32 season maps, 94.6% val_acc, val_loss 0.1722 |
| Adapter | 208,423 of 1,000,000 operations at its worst reference case |
| Inference | 50.39 ms at the worst reference case — **5.0%** of the 1000 ms a seat gets |
| Operators | Cast, Concat, Constant, Conv, Relu, Slice |
| Engine | `sha256:cd656bc84c1accec25106c035dece8cdac211e1e0f1a5aa6a2e7836c1cfc3bb4` |
| Model hash | `sha256:256f678ac080c397d46c9d9c92f43a3c5fb0b3efb75d06d672fe10bff646cc41` |
| Manifest hash | `sha256:70004d3f77a43cc0258f9dd51ffed81101a5284b67dff3e9079a4e36aa7c6223` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap. It is here because the turn deadline is what a graph too expensive to
play runs into, and a seat's share of it is the WHOLE turn -- one `model_infer` call per seat,
each with its own deadline.

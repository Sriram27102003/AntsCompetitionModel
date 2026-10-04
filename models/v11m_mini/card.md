# v11m_mini

A mini-class Ants policy.

| | |
|---|---|
| Weight class | **mini** — 1,021,459 of 1,048,576 bytes (97% of the cap) |
| Parameters | 494,997 (fp16 initializers) |
| Architecture | `MiniTrunk + max memory`, receptive field **15 cells** each way (dilations [1, 2, 4, 4, 2, 1, 1]) |
| Method | FatAss v11m_mini: /Users/sriram/Documents/AntsCompitionModel/v9/runs/v11m/init.pt with own-ant collisions removed in the graph and undefended adjacent enemy hills taken (safe_policy.py) |
| Adapter | 208,423 of 1,000,000 operations at its worst reference case |
| Inference | 102.72 ms at the worst reference case — **10.3%** of the 1000 ms a seat gets |
| Operators | Add, And, ArgMax, Cast, Concat, Constant, ConstantOfShape, Conv, Equal, Gather, Greater, Max, Mul, Not, Or, ReduceMax, Relu, Shape, Slice, Where |
| Measured on | engine `sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1` |
| Model hash | `sha256:aba190112e97df4c764991dbb07b9c840798da8cddcacd7a1d9fdf3b96a11cda` |
| Manifest hash | `sha256:db00b81c8f5bdb7dd3af5e7ec7fce9c6738ff66183d77544f138028b762e3c33` |
| Memory | `2` planes of `i8` (food_seen, hill_foe_seen), 2 bytes a cell: 29,760 bytes on the largest board. The season's class has to allow it; `check` judged the round trip over 202 chained observations |



Reproduce with:

```sh
python safe_policy.py /Users/sriram/Documents/AntsCompitionModel/v9/runs/v11m/init.pt /Users/sriram/Documents/AntsCompitionModel/models/v11m_mini --take-hills --rounds 8
```

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap. It is here because the turn deadline is what a graph too expensive to
play runs into, and a seat's share of it is the WHOLE turn -- one `model_infer` call per seat,
each with its own deadline.

**Measured on** is the engine the numbers above were read against, not a pin to match. What a
starter plays is the `engine` in its `games.toml`, which moves with each ants release; these numbers
were true of the engine named here and are not re-measured for every release, because the model
does not change when the engine does. A number that has to be current is read by running
`tinybrains check` yourself.

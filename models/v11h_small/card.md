# v11h_small

A small-class Ants policy.

| | |
|---|---|
| Weight class | **small** — 1,063,636 of 8,388,608 bytes (13% of the cap) |
| Parameters | 516,085 (fp16 initializers) |
| Architecture | `MiniTrunk + max memory`, receptive field **15 cells** each way (dilations [1, 2, 4, 4, 2, 1, 1]) |
| Method | FatAss v11h_small: /Users/sriram/Documents/AntsCompitionModel/v9/runs/v11w/best.pt with own-ant collisions removed in the graph and undefended adjacent enemy hills taken (safe_policy.py) |
| Adapter | 208,423 of 1,000,000 operations at its worst reference case |
| Inference | 85.26 ms at the worst reference case — **8.5%** of the 1000 ms a seat gets |
| Operators | Add, And, ArgMax, Cast, Concat, Constant, ConstantOfShape, Conv, Equal, Gather, Greater, Max, Mul, Not, Or, ReduceMax, Relu, Shape, Slice, Where |
| Measured on | engine `sha256:819166e7918117a1cbbd67dc042e7368a076a2dd45cea5122683983912d591e1` |
| Model hash | `sha256:87190e2b25badab87f7dd53c5a571498712e8473b9512b11bc73a3256aa2e8ce` |
| Manifest hash | `sha256:65a8e46460453f75c5746eac996a70f0cc76867f6f951d76f60d7edeec372f30` |
| Memory | `2` planes of `i8` (food_seen, hill_foe_seen), 2 bytes a cell: 29,760 bytes on the largest board. The season's class has to allow it; `check` judged the round trip over 202 chained observations |



Reproduce with:

```sh
python safe_policy.py /Users/sriram/Documents/AntsCompitionModel/v9/runs/v11w/best.pt /Users/sriram/Documents/AntsCompitionModel/models/v11h_small --take-hills --rounds 8 --small
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

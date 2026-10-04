"""Drop the least-used channels of a MiniTrunk checkpoint so the export fits a smaller weight class.

    python trim.py runs/v11w/best.pt 94 runs/v11m/init.pt

A channel's use is the size of the weights that write it times the size of the weights that read
it. Each layer boundary is trimmed on its own. The result is a starting point: it is close to the
original, not equal to it, and is trained on before it is judged.
"""
import sys

import torch

src, width, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["trunk"]
convs = sorted((k for k in sd if ".convs." in k and k.endswith(".weight")), key=lambda k: int(k.split(".")[-2]))
chain = convs + [k for k in sd if k.endswith("mix.weight")]          # layers that write `channels` planes
head = next(k for k in sd if k.endswith("head.weight"))
readers = chain[1:] + [head]
for w_key, r_key in zip(chain, readers):
    w, r = sd[w_key], sd[r_key]
    use = w.flatten(1).norm(dim=1) * r.transpose(0, 1).flatten(1).norm(dim=1)
    keep = use.argsort(descending=True)[:width].sort().values
    sd[w_key] = w[keep]
    b_key = w_key[:-6] + "bias"
    sd[b_key] = sd[b_key][keep]
    sd[r_key] = r[:, keep]
torch.save(ck, out)
print({k: tuple(v.shape) for k, v in sd.items() if v.dim() > 1})

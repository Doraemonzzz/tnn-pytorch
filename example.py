import torch
import torch.nn as nn

from tnn_pytorch import Gtu, TnnLayer, Tno

# batch size
b = 2
# number of head
h = 1
# sequce length
n = 10
# embedding size
e = 4
# rpe embedding size
d = 16

print("======Start Test Tno=====")
x = torch.rand(b, h, n, e)
models = [
    Tno(h, e, d, use_decay=True),
    Tno(h, e, d, use_multi_decay=True),
    Tno(h, e, d, causal=True),
]

for dim in [-2]:
    for model in models:
        y1 = model.forward(x, dim=dim)
        y2 = model.toeplizt_matrix(x, dim=dim)
        print(torch.norm(y1 - y2))
print("======End Test Tno=====")

print("======Start Test Gtu=====")
x = torch.rand(b, n, e)
models = [
    Gtu(
        embed_dim=e,
        num_heads=1,
    )
]

for dim in [-2]:
    for model in models:
        y = model(x)
        print(f"input size is {x.shape}")
        print(f"output size is {y.shape}")
print("======End Test Gtu=====")

print("======Start Test Tnn Layer=====")
x = torch.rand(b, n, e)
models = [
    TnnLayer(
        dim=e,
        num_heads=1,
        rpe_embedding=d,
        glu_dim=e,
    )
]

for dim in [-2]:
    for model in models:
        y = model(x)
        print(f"input size is {x.shape}")
        print(f"output size is {y.shape}")
print("======End Test Tnn Layer=====")

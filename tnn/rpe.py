import torch.nn as nn

from .helpers import ActLayer, get_activation_fn, get_norm_fn, print_params


class Rpe(nn.Module):
    def __init__(
        self, 
        dim, 
        outdim, 
        residual, 
        act="relu", 
        bias=True, 
        layers=3, 
        norm_type="simplermsnorm",
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.residual = residual
        self.outdim = outdim
        self.pos_dim = dim
        self.act = act
        self.pos_proj = nn.Linear(1, self.pos_dim, bias=bias)
        self.layers = nn.ModuleList([])
        for i in range(layers):
            self.layers.append(
                nn.Sequential(
                    get_norm_fn(norm_type)(self.pos_dim),
                    self.get_act(),
                    nn.Linear(self.pos_dim, self.pos_dim, bias=bias),
                )
            )
        self.out = nn.Sequential(
            get_norm_fn(norm_type)(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.outdim, bias=bias),
        )
        
    def get_act(self):
        if self.act == "silu":
            return nn.SiLU(inplace=True)
        elif self.act == "relu":
            return nn.ReLU(inplace=True)
        else:
            return ActLayer(self.act)

    def forward(self, biases):
        x = self.pos_proj(biases)
        if self.residual:
            for m in self.layers:
                x = m(x) + x
        else:
            for m in self.layers:
                x = m(x)
        x = self.out(x)

        return x

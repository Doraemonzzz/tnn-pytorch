import torch.nn as nn

from .glu import GLU
from .gtu import Gtu
from .helpers import get_norm_fn


class TnnLayer(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads,
        rpe_embedding,
        glu_dim,
        # model params
        prenorm=True,
        norm_type="simplermsnorm",
        # gtu params
        causal=False,
        gtu_act="silu",
        expand_ratio=3,
        use_decay=False,
        gamma=0.999,
        # rpe params
        rpe_act="relu",
        rpe_layers=3,
        # glu params
        glu_act="silu",
    ):
        super().__init__()
        self.token_mixer = Gtu(
            # gtu params
            embed_dim=dim,
            num_heads=num_heads,
            act_fun=gtu_act,
            norm_type=norm_type,
            causal=causal,
            expand_ratio=expand_ratio,
            use_decay=use_decay,
            gamma=gamma,
            # rpe params
            rpe_embedding=rpe_embedding,
            rpe_act=rpe_act,
            rpe_layers=rpe_layers,
        )

        self.token_norm = get_norm_fn(norm_type)(dim)
        self.feature_norm = get_norm_fn(norm_type)(dim)
        
        self.feature_mixer = GLU(
            d1=dim, 
            d2=glu_dim,
            act_fun=glu_act,
        )
        
        if prenorm:
            self.forward = self.forward_prenorm
        else:
            self.forward = self.forward_postnorm

    def forward_postnorm(self, x):
        x = x + self.token_norm(self.token_mixer(x))
        x = x + self.feature_norm(self.feature_mixer(x))

        return x
    
    def forward_prenorm(self, x):
        x = x + self.token_mixer(self.token_norm(x))
        x = x + self.feature_mixer(self.feature_norm(x))

        return x

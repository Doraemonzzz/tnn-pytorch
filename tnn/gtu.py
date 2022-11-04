import torch.nn as nn
from einops import rearrange

from .helpers import get_activation_fn, get_norm_fn, print_params
from .tno import Tno


class Gtu(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        act_fun="silu",
        causal=False,
        expand_ratio=3,
        resi_param=False,
        use_norm=False,
        norm_type="simplermsnorm",
        use_decay=False,
        use_multi_decay=False,
        rpe_layers=3,
        rpe_embedding=512,
        rpe_act="relu",
        normalize=False,
        par_type=1,
        residual=False,
        gamma=0.99,
        act_type="none",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.expand_ratio = expand_ratio
        self.resi_param = resi_param
        self.num_heads = num_heads
        self.normalize = normalize
        
        if self.resi_param:
            self.d = nn.Parameter(torch.randn(embed_dim))

        d1 = int(self.expand_ratio * embed_dim)
        d1 = (d1 // self.num_heads) * self.num_heads
        self.head_dim = d1 // num_heads
        # linear projection
        self.v_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.u_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.o = nn.Linear(d1, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)
        # tno
        self.toep = Tno(
            h=num_heads, 
            dim=self.head_dim,
            rpe_dim=rpe_embedding, 
            causal=causal, 
            use_decay=use_decay, 
            use_multi_decay=use_multi_decay,
            residual=residual,
            act=rpe_act,
            par_type=par_type,
            gamma=gamma,
            bias=bias,
            act_type=act_type,
            layers=rpe_layers,
            norm_type=norm_type,
        )
        # norm
        self.norm_type = norm_type
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm_fn(self.norm_type)(d1)
    
    def forward(self, x):
        # x: b, h, w, d
        num_heads = self.num_heads

        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'b n (h d) -> b h n d', h=num_heads)
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = u * output
        if self.use_norm:
            output = self.norm(output)
            
        output = self.o(output)
        
        return output


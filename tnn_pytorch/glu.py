import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import get_activation_fn, print_params


class GLU(nn.Module):
    def __init__(self, d1, d2, act_fun, fina_act="None", dropout=0.0, bias=True):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.l1 = nn.Linear(d1, d2, bias=bias)
        self.l2 = nn.Linear(d1, d2, bias=bias)
        self.l3 = nn.Linear(d2, d1, bias=bias)
        self.act_fun = get_activation_fn(act_fun)
        self.p = dropout
        if self.p > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        self.fina_act = get_activation_fn(fina_act)

    def forward(self, x):
        o1 = self.l1(x)
        weight = self.act_fun(o1)
        if self.p > 0.0:
            weight = self.dropout(weight)
        o2 = self.l2(x)
        output = weight * o2
        output = self.l3(output)
        output = self.fina_act(output)

        return output

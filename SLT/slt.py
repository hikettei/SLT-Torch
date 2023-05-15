
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math

from .maddness_legacy import (
    MaddnessMatmul
)

class IntermidateEmbedding():
    def __init__(self, embedding_dim, nheads, C=7):
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        
        prototype_length = embedding_dim/nheads/C
        self.C = embedding_dim / prototype_length
        self.maddness = None
        self.reset_state()
        
    def reset_state(self):
        self.maddness = MaddnessMatmul(C=self.C)

    def set_target(self):
        # Construct LUT
        pass

    def apply_matmul(self, source):
        pass

# Ref: https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/model.py
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class CachingMHA():
    def __init__(self, embedding_dim, maxlen, nheads):
        self.intermidate = IntermidateEmbedding(embedding_dim, nheads)

        assert n_state % config.n_head == 0
        self.n_head = nheads
        self.split_size = embedding_size
        self.c_attn = Conv1D(embedding_dim * 3, embedding_dim)
        self.c_proj = Conv1D(embedding_dim, embedding_dim)


    def forward(self, q, k, v, mask):
        pass

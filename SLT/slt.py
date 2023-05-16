
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math

from .maddness_legacy import (
    MaddnessMatmul
)

import time
from tqdm import tqdm

#Linear, matmul*2
class IntermidateEmbedding():
    def __init__(self, embedding_dim, nheads, C=7):
        self.embedding_dim = embedding_dim
        
        prototype_length = embedding_dim/nheads/C
        self.C = embedding_dim / prototype_length
        self.maddness = None
        self.reset_state()
        self.lock = False
        
    def reset_state(self):
        self.lock = False
        self.maddness = MaddnessMatmul(C=self.C)

    def set_target(self):
        # Construct LUT
        # Training from:
        # SourceとTargetをよく分類するMaddness Binary_hashing_tree
        # nsplits=4で固定、 Positional Encodnigを明示的に与えて、Bucketに加算 (Embedding(vocab_size * relative_position_candidates* ...))
        pass

    def apply_matmul(self, source):
        pass

class MProj():
    def __init__(self, embedding_dim, C=7):
        self.embedding_dim = embedding_dim
        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        prototype_length = embedding_dim/C
        self.C = embedding_dim / prototype_length
        self.C = 16
        self.maddness = None
        self.reset_state()

    def __call__(self, x, approx=False):
        if approx:
            return x
        else:
            return self.proj(x)

    def reset_state(self):
        self.maddness = MaddnessMatmul(C=self.C)

    def set_A_offline(self, a):
        if self.maddness is None:
            raise Exception("Call MProj.reset_state() first")

        self.maddness._learn_hash_buckets_and_prototypes(a)
        self.maddness._set_B(self.proj.weight.to('cpu').detach().numpy())

#Ref: https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/model.py

class CachingMHA():
    def __init__(self, embedding_dim, maxlen, nheads, C=7):
        self.intermidate = IntermidateEmbedding(embedding_dim, nheads, C=C)
        self.maxlen = maxlen
        
        assert embedding_dim % nheads == 0
        
        self.n_head = nheads
        self.split_size = embedding_dim
        
        self.q_proj = MProj(embedding_dim, C=C)
        self.k_proj = MProj(embedding_dim, C=C)
        self.v_proj = MProj(embedding_dim, C=C)

        self.out_proj = MProj(embedding_dim, C=C)

        self.reset_state()
        
    def _scal_dot_attn(self, q, k, v, mask=None):
        attn_w = torch.matmul(q, k)
        if mask is not None:
            attn_w = attn_w.data.masked_fill_(mask, -torch.finfo(torch.float).max)
        
        attn_w = attn_w / attn_w.size(-1)
        attn_w = nn.Softmax(dim=-1)(attn_w)

        res = torch.matmul(attn_w, v)

        return res

    def reset_state(self):
        self.maddness_qkv_proj_trained_p = False
        [layer.reset_state() for layer in [self.q_proj, self.k_proj, self.v_proj]]
        
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states
    
    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def train_g(self, cond, layers, training_data, approx=False):
        if not cond and approx:
            print("Training Encoding Function...")
            training_data = training_data.reshape((-1, training_data.shape[-1])).to('cpu').detach().numpy()
            for layer in tqdm(layers):
                layer.set_A_offline(training_data)
            return True
        return cond
        
    def forward(self, source, target, mask, approx=False):
        self.maddness_qkv_proj_trained_p = self.train_g(
            self.maddness_qkv_proj_trained_p,
            [self.q_proj, self.k_proj, self.v_proj],
            target,
            approx=approx)
        q, k, v = self.q_proj(source, approx=approx), self.k_proj(target, approx=approx), self.v_proj(target, approx=approx)
        q, k, v = self.split_heads(q), self.split_heads(k, k=True), self.split_heads(v)

        if approx:
            o = v
        else:
            o = self._scal_dot_attn(q, k, v, mask=mask)

        o = self.merge_heads(o)
        o = self.out_proj(o, approx=approx)
        return o
    

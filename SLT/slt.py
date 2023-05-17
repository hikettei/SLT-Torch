
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

import numba

# 文章長が変わったらReset?

#Linear, matmul*2
class IntermidateEmbedding():
    def __init__(self, embedding_dim, nheads, C=32):
        self.embedding_dim = embedding_dim
        
        self.C = C#// nheads
        self.nheads = nheads
        self.maddness_qk = None
        self.maddness_wv = None
        self.trained_p   = False
        
        self.reset_state()

        self.apply_qk_out = None
        
    def reset_state(self):
        self.trained_p = False
        self.maddness_qk = [MaddnessMatmul(C=self.C) for _ in range(self.nheads)]
        #self.maddness_wv = [MaddnessMatmul(C=self.C) for _ in range(self.nheads)]

    def train_maddness_qk(self, q, k, verbose=True):
        # TODO: Embeddingからランダムにサンプルを持ってきて, それとTarget間のLUTを構築

        # Memo
        # MHA: [Linear[k].weight = torch.concat([source[:, 0], source[:, 1], ...])] where k = 0...len(target)
        
        for nth_head, maddness in enumerate(self.maddness_qk):
            # Source-target attention as self-attention.
            q_offline = q[:, nth_head, :, :].reshape((-1, q.size(-1))).to('cpu').detach().numpy() # Corresponds to source
            k_offline = k[:, nth_head, :, :].reshape((-1, k.size(-1))).to('cpu').detach().numpy() # Corresponds to target

            # Encoding_Time < Total_Timeを満たす条件の中の頻度で, binary-treeを再学習, prototypeを再度構築
            maddness._learn_hash_buckets_and_prototypes(k_offline)
            maddness._set_B(q_offline.T)

            # Measure Accuracy
            if verbose:
                pass
                # QとKは全く別のweightから得られたものでは？
                # Wを無視するためのLSH
                
                # TODO...

    def train_maddness_wv(self, w, v):
        pass # TODO (w, v is not constant)

    def apply_qk_mm(self, q, k):
        # q ... source
        # k ... target
        # in this case, we ignore q but use prototypes instead.

        # Caching
        apply_qk_out = torch.zeros((1, q.size(1), q.size(-2), k.size(-2)), dtype=torch.float32)
        for nth_head, maddness in enumerate(self.maddness_qk):
            k_enc = k[:, nth_head, :, :].reshape((-1, k.size(-1))).to('cpu').detach().numpy()
            q_enc = k[:, nth_head, :, :].reshape((-1, k.size(-1))).to('cpu').detach().numpy()
            maddness._set_A(k_enc)
            #maddness._set_B(q_enc.T)
            res = torch.from_numpy(
                maddness._calc_matmul(
                    maddness.A_enc,
                    maddness.luts,
                    maddness.offset,
                    maddness.scale,
                    M=k.size(-2)).T)
            apply_qk_out[:, nth_head] = res.mean(axis=-2).reshape((1, -1))
        return apply_qk_out

    
    def apply_wv_mm(self, w, v):
        return torch.matmul(w, v) # TODO 
    
    def scal_dot_attn(self, q, k, v, mask=None):
        #q.size() = [1, nheads, sentence_length, embedding_dim/nheads]
        # [1, 8, 100, 98]

        if not self.trained_p:
            self.train_maddness_qk(q, k)
        w = self.apply_qk_mm(q, k)

        if mask is not None:
            w = w.data.masked_fill_(mask, -torch.finfo(torch.float).max)
            
        w = w / math.sqrt(w.size(-1))
        w = nn.Softmax(dim=-1)(w)

        if not self.trained_p:
            self.train_maddness_wv(w, v)
            
        res = self.apply_wv_mm(w, v)

        if not self.trained_p:
            self.trained_p = True
            
        return res

class MProj():
    def __init__(self, embedding_dim, C=32):
        self.embedding_dim = embedding_dim
        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.C = C # TODO: Test C=32, 64, ...
        self.maddness = None
        self.reset_state()

    def __call__(self, x, approx=False):
        if approx:
            # Excepted: batch_size = 1 (TODO: Remove this constraints)
            self.maddness._set_A(x.reshape(-1, x.size(-1)).to('cpu').detach().numpy())
            return torch.from_numpy(self.maddness._calc_matmul(
                self.maddness.A_enc,
                self.maddness.luts,
                self.maddness.offset,
                self.maddness.scale)).reshape(x.size())
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
    def __init__(self, embedding_dim, maxlen, nheads, C=32):
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
        self.maddness_out_proj_trained_p = False
        self.out_proj.reset_state()
        
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states
    
    def split_heads(self, x, k=False, approx=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k and not approx:
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
        assert source.size(0) == 1 and target.size(0) == 1, "CachingMHA: Set Batch_Size = 1"
        self.maddness_qkv_proj_trained_p = self.train_g(
            self.maddness_qkv_proj_trained_p,
            [self.q_proj, self.k_proj, self.v_proj],
            source,
            approx=False)
        
        q, k1, v = self.q_proj(source, approx=False), self.k_proj(target, approx=False), self.v_proj(target, approx=False)
        q, k, v  = self.split_heads(q), self.split_heads(k1, k=True, approx=approx), self.split_heads(v)

        if approx:
            o = self.intermidate.scal_dot_attn(q, k, v, mask=mask)
        else:
            o = self._scal_dot_attn(q, k, v, mask=mask)

        o = self.merge_heads(o)

        if not self.maddness_out_proj_trained_p and False:#approx:
            # Note: when to call train_g?
            self.maddness_out_proj_trained_p = True
            o_not_masked = self._scal_dot_attn(q, self.split_heads(k1, k=True, approx=False), v, mask=None)
            o_not_masked = self.merge_heads(o_not_masked)
            self.out_proj.set_A_offline(o_not_masked.reshape((-1, o_not_masked.size()[-1])).to('cpu').detach().numpy())
        
        o = self.out_proj(o, approx=False) # Fix approx=False?
        return o
    

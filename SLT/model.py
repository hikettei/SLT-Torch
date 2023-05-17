
import torch
import torch.nn as nn

from .SLT.maddness_legacy import (
    MaddnessMatmul
    )

import numpy as np

class SaltConfig():
    def __init__(self,
                 embedding_dim=784,
                 vocab_size=5000,
                 pad_idx=0,
                 nheads=12,
                 C=16,
                 use_embedding=None,
                 positional_encoder="RNN",
                 dropout=None):
        
        assert embedding_dim % nheads == 0
        
        self.embedding_dim = vocab_size
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.nheads = nheads
        self.C = C
        self.use_embedding = use_embedding # None or torch.Tensor
        self.positional_encoder = positional_encoder
        self.dropout = dropout

# Temporary
# __init__使えますか?
class SparseMatmul4D(nn.autograd.Function):
    def __init__(self, salt_embedding):
        self.salt_embedding = salt_embedding

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x)
        ctx.save_for_backward(y)
        return torch.matmul(x, y.T)

    @staticmethod
    def backward(ctx, dy):
        return torch.matmul(ctx.x, dy), torch.matmul(dy, ctx.y)


class SaltEmbedding(nn.Module):
    def __init__(self, config: SaltConfig):
        super(SaltEmbedding, self).__init__()
        self.embedding = nn.Embedding(config.embedding_dim, config.vocab_size, pad_idx=config.pad_idx)
        self.C         = config.C * config.nheads
        self.maddness  = MaddnessMatmul(C=self.C)
        self.config    = config

        if config.use_embedding is not None:
            self.load_embedding(config.use_embedding)
        
        self.optimize_embedding()

    def load_embedding(self, x):
        self.embedding.weight = x

    def optimize_embedding(self):
        # Obtain LSH, and each centroids.
        self.maddness._learn_hash_buckets_and_prototypes(self.embedding.weight)
        # Construct LUT

    def forward(self, x):
        """
        Return:
          Embeddings - [batch_size, sentence_length, embedding_dim] (no positional encoding)
        """
        return self.embedding(x)


def time_attention(positional_encoder, salt_embedding, q, k):
    """
    To -> (RNN) -> To+1 -> To+2 ...

    Input: (Q, K do not have positional informations.)
    Query   : I have a new pen...
    Keyword : So    I    do.
             So+1  I+1  do+1 (hidden_state1)
             So+2  I+2  do+2 (hidden_state2)

    That is:
              So    I    do
               I   do    have
              do   have   a    (matmul is working in each hidden_state).
    
    Assume that each hidden_state can be reconstructed by SaltEmbedding.weight's all prototypes.

    q [batch_size, nheads, time, embedding_dim//nheads]
    k [batch_size, nheads, time, embedding_dim//nheads]
    """

    whole_time = q.size(2)

    def apply_time_attn(nth_head):
        qn = q[:, nth_head, :, :].squeeze(1) # [batch_size, time, embedding_dim//nheads]
        kn = k[:, nth_head, :, :].squeeze(1) # [batch_size, time, embedding_dim//nheads]

        for t in range(whole_time):
            pass
        
    time_weights = torch.concat([apply_time_attn(nth) for nth in range(q.size(1))], dim=1)

    return time_weights

def merge_attn(positional_encoder, salt_embedding, q, k):
    """
    Merge Attentions:
      1. Semantical Attention. Just computing matmul with no positional embedded embeddings.
      2. Grammar Attention. Attention by Time-Weighting. (Kinda RNN)

    s_attn = softmax(Q@K/sqrt(dim))

    Inputs:
      query   (Input Source) [batch_size, nheads, seq_len, embedding_dim//nheads]
      keyword (Input Target) [batch_size, nheads, seq_len, embedding_dim//nheads]
    """
    assert q.size() == k.size()

    def apply_merge_attn(nth_head):
        # Computes semantic_attention by nth_head=0, 1, 2, ...
        # [batch_size, 1, seq_len, embedding_dim//nheads]
        q_n = q[:, nth_head, :, :]
        k_n = k[:, nth_head, :, :]
        return SparseMatmul4D()(q_n, k_n) # The equivalent to q_n @ k_n.T, returing [batch_size, 1, seq_len, seq_len]
    
    # Compute with Maddness(Sparse)
    semantic_w = torch.concat([apply_merge_attn(nth_head) for nth_head in range(q.size(1))], dim=1) # [batch_size, nheads, seq_len, seq_len]

    # Attention by Time.
    grammar_w  = time_attention(positional_encoder, salt_embedding, q, k)

    return semantic_w, grammar_w

class SaltMHA(nn.Module):
    def __init__(self, salt_embedding, config):
        super(SaltMHA, self).__init__()

        self.salt_embedding = salt_embedding

        self.config = config
        self.embedding_dim = config.embedding_dim
        self.nheads        = config.nheads

        if config.positional_encoder = "RNN":
            self.encoder = nn.RNN(config.embedding_dim//config.nheads,
                                  config.embedding_dim//config.nheads,
                                  num_layers=1,
                                  bias=True,
                                  batch_first=True,
                                  dropout=config.dropout) # [batch_size, seq_len, dim]
        else:
            raise Exception("Choose config.positional_encoder from: RNN")
        
    def split_heads(self, x):
        """
        Input:   x  [batch_size, seq_len, embedding_dim]
        Return: out [batch_size, nheads, seq_len, embedding_dim//nheads].
        """
        pass
        
    def forward(self, source, target):
        w = merge_attn(self.encoder, self.salt_embedding, source, target)

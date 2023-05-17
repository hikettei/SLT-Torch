
import torch
import torch.nn as nn

from .maddness_legacy import (
    MaddnessMatmul
)

import numpy as np

class SaltConfig():
    def __init__(self,
                 embedding_dim=784,
                 vocab_size=500,
                 pad_idx=0,
                 nheads=8,
                 C=16,
                 use_embedding=None,
                 positional_encoder="orthogonal",
                 dropout=None,
                 maddness=True):
        
        assert embedding_dim % nheads == 0
        
        self.embedding_dim = vocab_size
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.nheads = nheads
        self.C = C
        self.use_embedding = use_embedding # None or torch.Tensor
        self.positional_encoder = positional_encoder
        self.dropout = dropout
        self.maddness = maddness
        

# Temporary
# __init__使えますか?
"""
class SparseMatmul4D(nn.autograd.Function):
    def __init__(self, salt_embedding):
        self.salt_embedding = salt_embedding

    @staticmethod
    def forward(ctx, x, y, encoder=None):
        # forwardはMaddnessで、backwardは通常に？
        ctx.save_for_backward(x)
        ctx.save_for_backward(y)
        return torch.matmul(x, y.T)

    @staticmethod
    def backward(ctx, dy):
        return torch.matmul(ctx.x, dy), torch.matmul(dy, ctx.y), None
"""

class SaltEmbedding(nn.Module):
    def __init__(self, config: SaltConfig):
        super(SaltEmbedding, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_idx)
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
        self.maddness._learn_hash_buckets_and_prototypes(self.embedding.weight.detach().numpy())
        # Construct LUT

        # Encoding時にMHAを考慮するの忘れない

    def forward(self, x):
        """
        Return:
          Embeddings - [batch_size, sentence_length, embedding_dim] (no positional encoding)
        """
        return self.embedding(x)


def time_attention(positional_encoder, salt_embedding, q, k, decay_rate=0.0):
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
    positional_encoder [1, nheads, 1, embedding_dim//nheads]
    """

    whole_time = q.size(2)

    def apply_time_attn(nth_head):
        qn = q[:, nth_head, :, :].squeeze(1) # [batch_size, time, embedding_dim//nheads]
        kn = k[:, nth_head, :, :].squeeze(1) # [batch_size, time, embedding_dim//nheads]
        pe_w = positional_encoder[:, nth_head, :, :].squeeze(1) # [batch_size, 1, embedding_dim//nheads]

        # Compare QN [batch_size, whole_time, dim] vs [batch_size, t=0, dim], [batch_size, t=1, dim] ...

        # t=0
        #        Source             Target
        #  I have a new pen ... vs    So.
        #  I have a new pen ... vs  So, So[t+1], So[t+2], So[t+3], ...
        #
        # t=1
        #  I have a new pen ... vs  So, Do, Do[t+1], Do[t+2],      ... 
        # t=2
        #  I have a new pen ... vs  So, Do, I, I[t+1], I[t+2],     ...
        #
        # t=0~t=current_processing_time = accumlated in cumulative_context. and being reused.
        #
        # The total result is normalized

        # should cumulative_context be trainable? (k.weight works as initial-weight i guess)
        
        w_ret = torch.zeros((qn.size(0), qn.size(1), qn.size(1))) # [batch_size, time, time]
        
        cumulative_context = torch.zeros((kn.size(0), 1, kn.size(-1))) # [batch_size, 1, embedding_dim//nheads]
        cumulative_context[:, 0, :] += kn[:, 0, :]
        
        for t in range(1, whole_time+1):
            # Here, we predict Words t=n+1 from t=n and compute matmul.
            # [batch_size, time, embedding_dim//nheads] @ [batch_size, 1, embedding_dim//nheads].T

            # 自信ない・・・
            cumulative_context = nn.Tanh()(
                torch.mul(cumulative_context, pe_w.T) # [1 dim] * [1 dim] -> [1 dim]. t=0~t-1, * weight, (Q. Add bias?)
            )
            
            #cumulative_context = nn.Dropout()
            # qn [batch_size, time, dim] @ cumulative_context[batch_size, 1, dim].T
            print(qn.shape)
            print(cumulative_context.shape)
            w_ret[:, :, t-1] += torch.matmul(qn, cumulative_context.transpose(-2, -1)) # w_t = [batch_size, time, 1]

            # print(cosine_simirality(kn[:, t, :], cumulative_context)
            # Update cumulative_context for next iteration
            cumulative_context = torch.mul(kn[:, t, :], cumulative_context * (1.0 - decay_rate))
        return w_ret.sum(axis=1) / math.sqrt(q.size(-1)) # Return: [batch_size, 1, time]
        
    time_weights = torch.concat([apply_time_attn(nth) for nth in range(q.size(1))], dim=1)
    return nn.Softmax(dim=-1)(time_weights)

def merge_attn(positional_encoder,
               salt_embedding,
               q,
               k,
               alpha=1.0,
               beta=1.0,
               gamma=2.0):
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

    def apply_semantic_attn(nth_head):
        # Computes semantic_attention by nth_head=0, 1, 2, ...
        # [batch_size, 1, seq_len, embedding_dim//nheads]
        q_n = q[:, nth_head, :, :]
        k_n = k[:, nth_head, :, :]
        return torch.matmul(q_n, k_n.transpose(-2, -1)) #SparseMatmul4D()(q_n, k_n) # The equivalent to q_n @ k_n.T, returing [batch_size, 1, seq_len, seq_len]
    
    # Compute with Maddness(Sparse)
    semantic_w = torch.concat([apply_semantic_attn(nth_head) for nth_head in range(q.size(1))], dim=1) # [batch_size, nheads, seq_len, seq_len]

    # Attention by Time.
    grammar_w  = time_attention(positional_encoder, salt_embedding, q, k)

    # what semantic_w to grammar_w is what Transformer to RNN. (merging global and super wide but low-precise attention, fast but small range attn)

    # Merging
    return (alpha * semantic_w + beta * grammar_w) / z

class SaltMHA(nn.Module):
    def __init__(self, salt_embedding, config):
        super(SaltMHA, self).__init__()

        self.salt_embedding = salt_embedding

        self.config = config
        self.embedding_dim = config.embedding_dim
        self.nheads        = config.nheads

        if config.positional_encoder == "orthogonal":
            self.encoder = nn.Parameter(nn.init.orthogonal_(
                torch.empty(1, config.nheads, 1, config.embedding_dim // config.nheads), gain=1))
            # [1, nheads, 1, dim]
        else:
            raise Exception("Choose config.positional_encoder from: orthogonal")

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(2.0))

        # self.scale, self.beta (for reconstructing into x ~ Embedding)
        
    def split_heads(self, x):
        """
        Input:   x  [batch_size, seq_len, embedding_dim]
        Return: out [batch_size, nheads, seq_len, embedding_dim//nheads].
        """
        new_x_shape = x.size()[:-1] + (self.nheads, x.size(-1) // self.nheads)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # [batch_size, nheads, seq_len, embedding_dim//nheads]

        
    def forward(self, source, target):
        """
        Input: [batch_size, seq_len, embedding_dim]
        """

        q = self.split_heads(source) # Linear Transform?
        k = self.split_heads(target) # Linear Transform?

        # Reconstruct Q, K?
        
        w = merge_attn(self.encoder, self.salt_embedding, q, k, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        return w

def test():
    config = SaltConfig()
    emb    = SaltEmbedding(config)
    mha    = SaltMHA(emb, config)
    x = torch.randn([10, 100, 784])
    print(mha(x, x))

test()

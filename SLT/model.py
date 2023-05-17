
import torch
import torch.nn as nn

from .maddness_legacy import (
    MaddnessMatmul
)

import numpy as np

import math

class SaltConfig():
    def __init__(self,
                 embedding_dim=784,
                 vocab_size=500,
                 pad_idx=0,
                 nheads=8,
                 nlayers=6,
                 C=16,
                 layer_norm_eps=1e-12,
                 dim_ffn=1024,
                 use_embedding=None,
                 positional_encoder="orthogonal",
                 dropout=None,
                 maddness=True):
        
        assert embedding_dim % nheads == 0
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.nheads = nheads
        self.C = C
        self.use_embedding = use_embedding # None or torch.Tensor
        self.positional_encoder = positional_encoder
        self.dropout = dropout
        self.maddness = maddness
        self.nlayers = nlayers
        self.layer_norm_eps= layer_norm_eps
        self.dim_ffn = dim_ffn
        

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
        self.maddness._set_B(self.embedding.weight.detach().numpy().T)
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
        qn = q[:, nth_head, :, :] # [batch_size, time, embedding_dim//nheads]
        kn = k[:, nth_head, :, :] # [batch_size, time, embedding_dim//nheads]
        pe_w = positional_encoder[:, nth_head, :, :] # [batch_size, 1, 1, embedding_dim//nheads]

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
        
        w_ret = [] # [[batch_size, time, 1] * time]
        cumulative_context = torch.zeros((kn.size(0), 1, kn.size(-1))) # [batch_size, 1, embedding_dim//nheads]
        cumulative_context += kn[:, 0, :].unsqueeze(1)

        for t in range(0, whole_time):
            # Here, we predict Words t=n+1 from t=n and compute matmul.
            # [batch_size, time, embedding_dim//nheads] @ [batch_size, 1, embedding_dim//nheads].T

            # is it ok?
            cumulative_context = nn.Tanh()(
                torch.mul(cumulative_context, pe_w) # [1 dim] * [1 dim] -> [1 dim]. t=0~t-1, * weight, (Q. Add bias?)
            )
            
            # cumulative_context = nn.Dropout()
            # qn [batch_size, time, dim] @ cumulative_context[batch_size, 1, dim].T
            out = torch.matmul(qn, cumulative_context.transpose(-2, -1)) # w_t = [batch_size, time, 1]
            w_ret.append(out)
            # print(cosine_simirality(kn[:, t, :], cumulative_context)
            # Update cumulative_context for next iteration
            cumulative_context = torch.mul(kn[:, t, :].unsqueeze(1), cumulative_context * (1.0 - decay_rate))
            # torch.concat(w_ret, dim=-1) -> [batch_size, time, time]
            # unsqueeze it -> [batch_size, 1, time, time]
        return torch.concat(w_ret, dim=-1).unsqueeze(dim=1) / math.sqrt(q.size(-1)) # Return: [batch_size, 1, time]
        
    time_weights = torch.concat([apply_time_attn(nth) for nth in range(q.size(1))], dim=1)
    # time_weights [batch_size, nheads, time]
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
        return torch.matmul(q_n, k_n.transpose(-2, -1)).unsqueeze(dim=1) #SparseMatmul3D()(q_n, k_n) # The equivalent to q_n @ k_n.T, returing [batch_size, 1, seq_len, seq_len]
    
    # Compute with Maddness(Sparse)
    semantic_w = torch.concat([apply_semantic_attn(nth_head) for nth_head in range(q.size(1))], dim=1) / 100.0 # [batch_size, nheads, seq_len, seq_len]

    # Attention by Time.
    grammar_w  = time_attention(positional_encoder, salt_embedding, q, k)

    # what semantic_w to grammar_w is what Transformer to RNN. (merging global and super wide but low-precise attention, fast but small range attn)

    # avoid unexcepted broadcasting
    assert semantic_w.shape == grammar_w.shape
    # Ensembling
    return nn.Softmax(dim=-1)((alpha * semantic_w + beta * grammar_w) / gamma)

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

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)
    
    def forward(self, source, target):
        """
        Input: [batch_size, seq_len, embedding_dim]
        """

        q = self.split_heads(source) # Linear Transform?
        k = self.split_heads(target) # Linear Transform?

        # Reconstruct Q, K?
        
        w = merge_attn(self.encoder, self.salt_embedding, q, k, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        w = self.merge_heads(w)
        return torch.matmul(w.transpose(-2, -1), target)

#Ref: https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/model.py
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

#Ref: https://zenn.dev/yukiyada/articles/59f3b820c52571#3.5-position-wise-feed-forward-networks
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(nn.functional.relu(self.linear1(x)))
    
class Block(nn.Module):
    def __init__(self, embedding_layer, config):
        super(Block, self).__init__()
        self.mha = SaltMHA(embedding_layer, config)
        self.ln1 = LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.ln2 = LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.ffn = FFN(config.embedding_dim, config.dim_ffn)
        
    def forward(self, x, y):
        attn = self.mha(self.ln1(x), self.ln2(y))
        y = y + self.ffn(y + attn)
        return y
    
class SaltGPT(nn.Module):
    def __init__(self, config):
        super(SaltGPT, self).__init__()
        self.embedding = SaltEmbedding(config)
        self.layers = nn.ModuleList([Block(self.embedding, config) for i in range(config.nlayers)])

    def forward(self, x, y):
        x_emb = self.embedding(x)
        y_emb = self.embedding(y)

        for layer in self.layers:
            y_emb = layer(x_emb, y_emb)

        return y_emb

def test():
    config = SaltConfig()
    model  = SaltGPT(config)
    print(model)
test()


import torch
import torch.nn as nn

from .maddness_legacy import (
    halut_encode_opt,
    MultiSplit,
    learn_proto_and_hash_function,
    split_lists_to_numpy
)

import numpy as np

import math

import pickle
import os

from tqdm import tqdm
import numba
from numba import prange

class MaddnessUtils():
    def __init__(self, C=16):
        self.C = C
        self.luts = None

    def set_lut(self, luts):
        self.luts = luts
        
    def learn_hash_buckets_and_prototypes(self, A):
        _, D = A.shape
        if D < self.C:
            raise Exception("D < C: {} < {}".format(D, self.C))
        self.splits_lists, self.prototypes, _ = learn_proto_and_hash_function(
            A, self.C, lut_work_const=1 # No Prototype optimizing.
        )

        self.ret_array, _, _ = split_lists_to_numpy(self.splits_lists)

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump([self.ret_array], f)

    def restore_model(self, path):
        with open(path, "rb") as f:
            self.ret_array, = pickle.load(f)
        
class SaltConfig():
    def __init__(self,
                 embedding_dim=784,
                 vocab_size=500,
                 pad_idx=0,
                 nheads=8,
                 nlayers=6,
                 C=16,
                 layer_norm_eps=1e-12,
                 dim_ffn=512,
                 use_embedding=None,
                 positional_encoder="orthogonal",
                 maddness_save_path="./maddness_tmp.pickle",
                 opt_forward=True,
                 opt_backward=True,
                 dropout=0.9,
                 maddness=True,
                 diffusion_step=10):
        
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
        self.maddness_save_path = maddness_save_path
        self.opt_forward = opt_forward
        self.opt_backward = opt_backward
        self.diffusion_step = diffusion_step

def construct_sparse_embedding(weight: np.ndarray, weight_enc: np.ndarray, C: int, K:int = 16):
    """
    weight     [vocab_size, embedding_dim]
    weight_enc [vocab_size, dim//C]
    """

    #assert weight.dims() == 2 and weight_enc.dims() == 2
    
    _, embedding_dim = weight.shape
    STEP = embedding_dim // C
    out_luts = np.zeros((C, K, K), dtype=np.float32) # [NPrototype, ncentroid, ncentroid]

    for cth in tqdm(range(0, C, STEP)):
        weight_c     = weight[:, cth:(cth+STEP)] # [vocab_size, STEP]
        weight_enc_c = weight_enc[:, cth//STEP]  # [vocab_size]

        # Aggregate = Geometric Mean

        #
        #     1 2 3 
        #   1 1 2 3
        #   2 2 4 6
        #   3 3 6 9 <- TODO: Remove duplicates for memory-efficiency, but for simplicity, ignore it for a while.
        #

        # Should be computed with Circular Mean...
        centroids_source = np.zeros((K, STEP), dtype=np.float32) # Dot Product / sqrt(dim)
        #centroids_target = np.ndarray((K, STEP), dtype=np.float32) # Dot Product / sqrt(dim)

        # Binary_Tree_Loss = SSE
        for kth in range(K):
            kplace = np.where(weight_enc_c == kth, True, False)
            k_cluster_weights = weight_c[kplace, :] # [:, STEP]
            centroids_source[kth] = k_cluster_weights.mean(axis=0)

        # Optimize Protos? Or compute loss by any measure?

        for source_kth in range(K):
            for target_kth in range(K):
                source = centroids_source[source_kth, :]
                target = centroids_source[target_kth, :]
                
                w = np.dot(source, target)
                out_luts[cth//STEP, source_kth, target_kth] = w
    return out_luts

@numba.jit(nopython=True, parallel=True)
def aggregate_enc(A_enc: np.ndarray, B_enc: np.ndarray, luts: np.ndarray):
    """
    A_enc [seq_len1, C]
    B_enc [seq_len2, C]

    luts [C, K, K]

    return
      - [seq_len1, seq_len2]
    """
    
    seq_len1 = A_enc.shape[0]
    seq_len2 = B_enc.shape[0]

    out = np.zeros((seq_len1, seq_len2), dtype=np.float32)
    C = A_enc.shape[1]

    for cth in prange(C):
        # Iteration By Prototypes
        A_enc_c = A_enc[:, cth] # TODO: Column-major
        B_enc_c = B_enc[:, cth] # TODO: Column-major

        for source_i in range(seq_len1):
            for target_i in range(seq_len2):
                out[source_i, target_i] += luts[cth, A_enc_c[source_i], B_enc_c[target_i]]

    return out
    

class SparseMatmul3dNode(torch.autograd.Function):
    @staticmethod
    def forward(x, y, encoder=None):
        # Encoder = SaltEmbedding

        # if not using maddness, this node is residual. so how's using uint8 only here.
        #out = torch.matmul(x, y.transpose(-2, -1))
        A_enc = encoder.diffusion.encode_state(x)
        B_enc = encoder.diffusion.encode_state(y)
        out = encoder.diffusion.apply(A_enc, B_enc)
        return out
    
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, y, _ = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, dy):
        x, y = ctx.saved_tensors
        return torch.matmul(dy.transpose(-2, -1), y), torch.matmul(x.transpose(-2, -1), dy).transpose(-2, -1), None

class SparseMatmul3D(nn.Module):
    def __init__(self, encoder):
        super(SparseMatmul3D, self).__init__()
        self.encoder = encoder

    def forward(self, x, y):
        """
        Input:
          x - [batch_size, N, D]
          y - [batch_size, M, D]
        """
        
        if self.encoder.config.opt_forward:
            return SparseMatmul3dNode.apply(x, y, self.encoder)
        else:
            return torch.matmul(x, y.transpose(-2, -1))
    
class SaltEmbedding(nn.Module):
    def __init__(self, config: SaltConfig):
        super(SaltEmbedding, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_idx)
        self.C         = config.C * config.nheads
        self.maddness  = MaddnessUtils(C=self.C)
        self.config    = config

        if config.use_embedding is not None:
            self.load_embedding(config.use_embedding)

        if config.opt_forward or config.opt_backward:
            self.optimize_embedding()

    def load_embedding(self, x):
        print(f"Vocab_Size: {x.size(0)} Embedding_dim: {x.size(1)}")
        self.embedding.weight = nn.Parameter(x)
        self.embedding.weight.requires_grad = False

    def optimize_embedding(self):
        if os.path.exists(self.config.maddness_save_path):
            print(f"Resumption from {self.config.maddness_save_path}")
            self.maddness.restore_model(self.config.maddness_save_path)
        else:
            print(f"Constructing binary-tree-splits and LUTs for Embedding.")
            self.maddness.learn_hash_buckets_and_prototypes(self.embedding.weight.detach().numpy()) # Adding noise?
            self.maddness.save_model(self.config.maddness_save_path)
            print(f"The result is saved at {self.config.maddness_save_path}")

        self.diffusion = IndexingDiffusion(self)
        weight_enc = self.diffusion.encode_state(self.embedding.weight.detach().unsqueeze(0))
        luts = construct_sparse_embedding(self.embedding.weight.to('cpu').detach().numpy(),
                                          weight_enc[0],
                                          self.C)
        self.maddness.set_lut(luts)

    def forward(self, x):
        """
        Return:
          Embeddings - [batch_size, sentence_length, embedding_dim] (no positional encoding)
        """
        out = self.embedding(x)
        return out
    
class IndexingDiffusion():
    def __init__(self, embedding: SaltEmbedding):
        self.maddness  = embedding.maddness
        self.embedding = embedding

    def encode_state(self, x: torch.Tensor) -> np.ndarray:
        """
        Encodes x (torch.Tensor) [batch_size, seq_len, dim] into out (np.ndarray, np.uint8)[batch_size, seq_len, dim // (C * nheads)]
        """
        out = np.zeros((x.size(0), x.size(1), self.maddness.C), dtype=np.uint32) #Column-Major [Batch-Size, STEP, N]

        # offsets = [0, 1, 2, 3, ...]
        # offsets = np.arange(self.maddness.C) * 16
        
        for n_batch in range(x.size(0)):
            out[n_batch] = halut_encode_opt(x[n_batch].to('cpu').detach().numpy(), self.maddness.ret_array)# + offsets

        # MaddnessHash: Loss -> Cumulative_CosSim
        # dim=784, nheads=8, C=7 (expt 16 14) = 72057594037927936
        # 同一のプロトタイプにのみLUTを構築する
        # P1 ... 16 * 16 = 256
        # P2 ... 16 * 16 = 256
        #        ...
        # LUT's size is that: 256 * C = 14336.
        return out

    def apply(self, x_enc, y_enc):
        out = np.zeros((x_enc.shape[0], x_enc.shape[1], y_enc.shape[1]), dtype=np.float32)
        for nth in range(x_enc.shape[0]):
            out[nth] = aggregate_enc(x_enc[nth], y_enc[nth], self.maddness.luts)

        return torch.from_numpy(out)


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

    q [batch_size, nheads, time1, embedding_dim//nheads]
    k [batch_size, nheads, time2, embedding_dim//nheads]
    positional_encoder [1, nheads, 1, embedding_dim//nheads]
    """

    whole_time = q.size(2)

    def apply_time_attn(nth_head):
        qn = q[:, nth_head, :, :] # [batch_size, time, embedding_dim//nheads]
        kn = k[:, nth_head, :, :] # [batch_size, time, embedding_dim//nheads]
        pe_w = positional_encoder[nth_head] # RNN

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
        # Todo: Mask Target When Training.
        # Todo: When Inferencing, Salt doesn't have to wait until the t+1 word predicted.
        # So looping Salt until the result gets enough good.
        # The total result is normalized

        # should cumulative_context be trainable? (k.weight works as initial-weight i guess)
        
        w_ret = [] # [[batch_size, time, 1] * time]

        next_word, hidden_state = pe_w(kn[:, 0, :].unsqueeze(1))

        # [batch_size, whole_time, dim] @ [batch_size, 1, dim]
        out = torch.matmul(qn, next_word.transpose(-2, -1))
        w_ret.append(out)

        for t in range(1, whole_time):
            # Here, we predict Words t=n+1 from t=n and compute matmul.
            # [batch_size, time, embedding_dim//nheads] @ [batch_size, 1, embedding_dim//nheads].T

            # get cumulative_context[t+1]
            next_word, hidden_state = pe_w(next_word, hidden_state)
            # cumulative_context = nn.Dropout()
            # qn [batch_size, time, dim] @ cumulative_context[batch_size, hidden_dim, dim]
            
            out = torch.matmul(qn, next_word.transpose(-2, -1))
            w_ret.append(out)
            
            #next_word = nn.ReLU()(torch.mul(next_word, kn[:, t, :].unsqueeze(1)))

            #print(np.matmul(kn[:, t, :].detach().numpy(), cumulative_context.detach().numpy().transpose(-2, -1)))
            
        return torch.concat(w_ret, dim=-1).unsqueeze(dim=1) / math.sqrt(q.size(-1)) # Return: [batch_size, 1, time]
        
    time_weights = torch.concat([apply_time_attn(nth) for nth in range(q.size(1))], dim=1)
    # time_weights [batch_size, nheads, time]
    return time_weights#nn.Softmax(dim=-1)(time_weights)

def merge_attn(positional_encoder,
               salt_embedding,
               q,
               k,
               state):
    """
    Merge Attentions:
      1. Semantical Attention. Just computing matmul with no positional embedded embeddings.
      2. Grammar Attention. Attention by Time-Weighting. (Kinda RNN)

    s_attn = softmax(Q@K/sqrt(dim))

    Inputs:
      query   (Input Source) [batch_size, nheads, seq_len1, embedding_dim//nheads]
      keyword (Input Target) [batch_size, nheads, seq_len2, embedding_dim//nheads]
    """
    assert q.size() == k.size()

    def apply_semantic_attn(nth_head):
        # Computes semantic_attention by nth_head=0, 1, 2, ...
        # [batch_size, 1, seq_len, embedding_dim//nheads]
        q_n = q[:, nth_head, :, :]
        k_n = k[:, nth_head, :, :]
        
        return SparseMatmul3D(salt_embedding)(q_n, k_n).unsqueeze(dim=1) / math.sqrt(q_n.size(-1)) # The equivalent to q_n @ k_n.T, returing [batch_size, 1, seq_len, seq_len]

    assert state in ("semantic", "grammar")
    
    if state == "semantic":
        # Compute with Maddness(Sparse)
        w = torch.concat([apply_semantic_attn(nth_head) for nth_head in range(q.size(1))], dim=1) # / 100.0  # [batch_size, nheads, seq_len, seq_len]
    else:
        # Attention by Time.
        w  = time_attention(positional_encoder, salt_embedding, q, k)
        
    # avoid unexcepted broadcasting
    return nn.Softmax(dim=-1)(w / math.sqrt(w.size(-1)))

class SaltMHA(nn.Module):
    def __init__(self, salt_embedding, config, attn_type="semantic"):
        super(SaltMHA, self).__init__()

        # SaltMHA has a two state of MHA:
        # semantic ... MHA without PE
        # grammar  ... MHA with RNN
        
        assert attn_type in ("semantic", "grammar")

        self.state = attn_type
        
        self.salt_embedding = salt_embedding

        self.config        = config
        self.embedding_dim = config.embedding_dim
        self.nheads        = config.nheads

        size = config.embedding_dim // config.nheads
        
        if attn_type == "grammar":
            if config.positional_encoder == "orthogonal":
                size = config.embedding_dim // config.nheads
                self.encoder = nn.ModuleList([
                    nn.RNN(size, size, batch_first=True) for i in range(config.nheads)
                ])
            else:
                raise Exception("Choose config.positional_encoder from: orthogonal")
        else:
            self.encoder = None
            self.W_k = nn.Parameter(torch.randn([1, config.nheads, size, size]) * math.sqrt(2 / size + size))

        self.W_q = nn.Parameter(torch.randn([1, config.nheads, size, size]) * math.sqrt(2 / size + size))
        self.W_v = nn.Parameter(torch.randn([1, config.nheads, size, size]) * math.sqrt(2 / size + size))

        self.W_o = nn.Parameter(torch.randn([1, config.nheads, size, size]) * math.sqrt(2 / size + size))
        
        self.dropoutq = nn.Dropout(config.dropout)
        self.dropoutk = nn.Dropout(config.dropout)
        self.dropoutv = nn.Dropout(config.dropout)

        
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
        # TODO: Q, K, V
        
        q = self.split_heads(source)        
        k = self.split_heads(target)    
        v = self.split_heads(target)

        
        q = torch.matmul(q, self.W_q)
        if self.state == "semantic":
            k = torch.matmul(k, self.W_k)
        v = torch.matmul(v, self.W_v)
        
        q, k, v = self.dropoutq(q), self.dropoutk(k), self.dropoutv(v)

        w = merge_attn(self.encoder, self.salt_embedding, q, k, self.state)
        
        out = torch.matmul(w, v) # w <- word[0] * weight[0] + word[1] * weight[1] * ...

        out = torch.matmul(out, self.W_o)
        # Out Lienar?
        return self.merge_heads(out)

#Ref: https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/model.py
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
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
        self.rnn = SaltMHA(embedding_layer, config, attn_type="grammar")
        self.mha = SaltMHA(embedding_layer, config, attn_type="semantic")
        
        self.ln1 = LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.ln2 = LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.ln3 = LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        
        self.ffn1 = FFN(config.embedding_dim, config.dim_ffn)
        self.ffn2 = FFN(config.embedding_dim, config.dim_ffn)
        self.ffn3 = FFN(config.embedding_dim, config.dim_ffn)
        

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        
    def forward(self, x, y):
        
        # Convection Step
        y = y + (1/3) * self.dropout1(self.ffn1(self.ln1(y)))

        # Diffusion Step
        y_attn = self.rnn(x, y)

        # Convection Step
        y = y + (1/3) * self.dropout2(self.ffn2(self.ln2(y_attn + y)))

        # Diffusion Step
        y_attn = self.mha(x, y)
        
        y = y + (1/3) * self.dropout3(self.ffn3(self.ln3(y_attn + y)))
        return y

class SaltGPT(nn.Module):
    def __init__(self, config):
        super(SaltGPT, self).__init__()
        self.embedding = SaltEmbedding(config)
        self.layers = nn.ModuleList([Block(self.embedding, config) for i in range(config.nlayers)])
        self.linear_out = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, x, y, y_past=None, is_last=False):
        x_emb = self.embedding(x)
        y_emb = self.embedding(y) if y_past is None else y_past

        for layer in self.layers:
            y_emb = layer(x_emb, y_emb)

        if is_last:
            return self.linear_out(y_emb)
        else:
            return y_emb

def step_model(config, model, x, y):
    """
    
    """
    y_past = None
    for n in range(config.diffusion_step-1):
        y_past = model(x, y, y_past=y_past)

    return model(x, y, y_past=y_past, is_last=True)

# TODO: Pickle maddness
def test():
    import time
    config = SaltConfig(nlayers=6, dim_ffn=512)
    #weights = torch.load('./gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)['wte.weight']
    #config.use_embedding = weights
    config.maddness_save_path = "./tmp1.pickle"
    #config.vocab_size = weights.size()[0]
    #config.embedding_dim = weights.size()[1]
    model  = SaltGPT(config)
    #model = torch.compile(model)
    x = torch.randint(0, config.vocab_size, (3, 300))
    y = torch.randint(0, config.vocab_size, (3, 300))
    t1 = time.time()
    out = model(x, y)
    t2 = time.time()
    print(f"{t2 - t1} sec (first iter)")
    print(torch.argmax(out, dim=-1))
    print(out.shape)
    print(((out - model.embedding(y)) ** 2).mean().backward())

    t1 = time.time()
    out = model(x, y)
    t2 = time.time()
    print(f"{t2 - t1} sec (second iter)")

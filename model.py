#  the goal of this LLaMa from scratch is to create the inference section so that the weights from the facebook can be downloaded and used,

import torch
import torch.nn.functional as F
import math 

from torch import nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    # according to the base version of LLaMa 2
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # heads for query as per the Grouped Multi-Query Attention
    n_kv_heads: Optional[int] = None # heads for the KV cache
    vocab_size: int = -1 # Later set in the build method

    # required for the FFN
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None # this means that the datatype of this variable can be either float or None

    # required for RMSNormalizaton
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    # CPU or GPU thing
    device: str = None


# Rotary Positional Encoding
# precompute all the combination of m (pos) & theta as per the paper
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):

    # head_dim : how much embedding data is in a single head 
    
    # as per the formula in the RoFormer
    theta_numerator = torch.arange(0, head_dim , 2).float()

    # (dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # for the 'm' parameter (position)
    m = torch.arange(seq_len, device = device) # actually seq_len * 2

    # output dot product to calculate all the combination of m and theta
    freqs = torch.outer(m, theta).float() # (seq_len, head_dim / 2)

    # now this needs to be transformed into the polar coordinate form
    # (m1.theta1) -> (cos(m1.theta1) + i x sin(m1.theta1))
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):

    # two consecutive tokens_embs will be a single complex number:
    # x1, x2 -> x1 + i*x2 and so on
    # (Batch_size, Seq_len, H, Head_Dim) -> (Batch_size, Seq_len, H, Head_Dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # reshaping to match the dimension of x_complex for element wise multiplication
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)


    # (Batch_size, Seq_len, H, head_dim / 2) * (1, seq_len, 1, head_dim / 2) => (Batch_size, seq_len, H, Head_dim / 2)
    x_rotated = x_complex * freqs_complex

    # convert the complex number back to the real number
    # (batch_size, seq_len, H, head_dim / 2) -> (batch_size, seq_len, H, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    
    #(batch_size, seq_len, h, head_dim /2, 2) -> (batch_size, seq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # gamma as learnable parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        #(batch_size, seq_len, dim) * (batch_size, seq_len, 1) => (batch_size, seq_len, dim)
        # rsqrt = 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # broadcasting i suppose
        # (dim) * (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # normalization before the attention block
        self.attention_norm = RMSNorm(args.dim, eps = args.norm_eps)

        # normalization after the attention block 
        self.ffn_norm = RMSNorm(args.dim, eps = args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):

        #(batch_size, seq_len, dim) + (batch_size, seq_len, dim) => (batch_size, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)

        #(batch_size, seq_len, dim) + (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

# the infamous grouped multi-query self-attention with KV cache
class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        # number of heads for the key and value (grouped multi-query)
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        # multi-query part
        self.n_heads_q = args.n_heads

        # times that n_kv_heads should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # dimension of each head (4096 / 32)
        self.head_dim = args.dim // args.n_heads 

        # learnable parameters
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias = False)
        # for KV cache implementation
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias = False)

        # concatenation head weights
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias = False)

        # cache implementation for KV pairs
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch_size, seq_len, n_kv_heads, head_dim = x.shape

        if n_rep == 1:
            return x
        
        return (
            # (batch_size, seq_len, N_KV_HEADS, 1, HEAD_DIM
            x[:, :, :, None, :]
            # (batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            # (batch_size, seq_len, n_kv_heads * n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):

        '''
            1. Pass through the learned weights Wq, Wk, Wv
            2. Reshape each xq, xk and xv into multi-query heads
            3. Apply rotary positional embedding to Query and Key
            4. Update the Key and Value cache according to new xk and xv
            5. Take the Key and Value for 'T' step as required
            6. Repeat the Key and Value as per GQA, so that the number of heads of Q matches to that of K and V heads
            7. Transform the matrics as per required matrix multiplication dimension needs
            8. Apply self-attention matmul function with Q, K transpose with values
            9. Concatenate the output from all heads (32) into a single tensor and apply the learned weight Wo
        '''

        # (batch_size, 1, dim) for the inference seq_len = 1
        batch_size, seq_len, _ = x.shape
        
        xq = self.wq(x) # (batch_size, 1, Dim) -> (batch_size, 1, H_Q * Head_Dim)
        xk = self.wk(x) # (batch_size, 1, Dim) -> (batch_size, 1, H_KV * Head_Dim)
        xv = self.wv(x) # (batch_size, 1, Dim) -> (batch_size, 1, H_KV * Head_Dim)

        # reshape the matrices to the required multi-heads as per GQA
        # (batch_size, 1, H_Q * Head_dim) -> (batch_size, 1, H_Q, Head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (batch_size, 1, H_Q * Head_dim) -> (batch_size, 1, H_KV, Head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (batch_size, 1, H_Q * Head_dim) -> (batch_size, 1, H_KV, Head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # applying rotary positional embeddings to the Q and K
        xq = apply_rotary_embeddings(xq, freqs_complex, device = x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device = x.device)

        # update the cache with new xk and xv values
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # keys and value required for the 'T' step inference

        # (batch_size, seq_len_kv, h_kv, head_dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # since every group of Q shares same K and V so the 4 KV heads for duplicated such that it matches 8 heads of the query, such that each 2 heads of the query takes in a single KV head
        # (batch_size, seq_len, H_KV, Head_dim) -> (batch_size, seq_len, H_Q, Head_dim)
        keys = self.repeat_kv(keys, self.n_rep)
        values = self.repeat_kv(values, self.n_rep)

        # (batch_size, 1, H_Q, Head_dim) -> (batch_size, H_Q, 1, Head_dim)
        xq = xq.transpose(1, 2)

        # (batch_size, 1, H_Q, Head_dim) -> (batch_size, H_Q, 1, Head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_dim) . (B, H_Q, Head_dim, seq_len_kv) -> (B, H_Q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim = -1).type_as(xq)

        # (B, H_Q, 1, 1) @ (B, H_Q, 1, Head_dim) -> (B, H_Q, 1, Head_dim)
        output = torch.matmul(scores, values)

        # (B, H_Q, 1, Head_dim) -> (B, 1, H_Q, Head_dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
            self, args: ModelArgs
    ):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, seq_len, dim) -> (B, seq_len, hidden_dim)
        swish = F.silu(self.w1(x))
        # (B, seq_len, dim) -> (B, seq_len, hidden_dim)
        x_v = self.w3(x)
        # out_dim -> (B, seq_len, hidden_dim)
        x = swish * x_v

        # (B, seq_len, hidden_dim) -> (B, seq_len, dim)
        x = self.w2(x)
        return x

# skeleton of the LLaMA Model

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size hasn't been set yet."

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim) # (vocab_size x d_model)

        # all the encoder block layers
        self.layers = nn.ModuleList()

        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps = args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias = False)

        # requisites for rotary positional encoding
        # self.args.max_seq_len * 2, cause we also have the prompt along with the input, so ??? 
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)

    # this LLaMa forward method is constructed for inference using the pre-trained model weights, so the seq_len is one
    def forward(self, tokens: torch.Tensor, start_pos: int):

        #(batch_size, seq_len: 1)
        batch_size, seq_len = tokens.shape

        assert seq_len == 1, "Only one token at a time is processed while generating the text"

        # (batch_size, seq_len) -> (batch_size, seq_len, token_dims)
        h = self.tok_embeddings(tokens)

        # calculate the (m, theta) for [start_pos, start_pos + seq_len: 1]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # apply all the encoder blocks 
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.output(h).float()
        return output
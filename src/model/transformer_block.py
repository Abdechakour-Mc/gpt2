import torch
import torch.nn as nn
import torch.nn.functional as f

import math


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * self.heads == embed_size
        ), "Embedding size must be divisible by number of heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0] # batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1] # Seq length

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Compute energy (Q*K)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries,keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # Apply softmax to get the attention scores
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out 
    

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # In case the embed size is an odd number
        if embed_size % 2 != 0:
          self.embed_size = embed_size + 1
        else:
          self.embed_size = embed_size

        # Creating a matrix of shape (max_len, embed_size) for Pos Encs
        pos_encoding = torch.zeros(max_len, self.embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_size, 2).float() * (-math.log(10000.0)/self.embed_size))

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        pos_encoding = pos_encoding.unsqueeze(0) # Batch dim
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        # Adding the pos to the input embds
        return x * 0 + self.pos_encoding[:, :x.size(1), :x.size(2)] 


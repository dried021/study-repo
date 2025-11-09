from torch import nn
import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arrange(0, max_len, dtype=torch.float).unsqeeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
            x: torch.Tensor [batch_size, seq_len, d_model]
            returns: torch.Tensor
        '''
        return x + self.pe[:, :x.size(1)]
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias = False)
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)
        self.w_o = nn.Linear(d_model, d_model, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def scaled_dot_product_attention(
            self, q, k, v, mask=None
    ):
        '''
            q, k, v: torch.Tensor [batch_size, n_heads, seq_len, d_k]
            mask: [batch_size, 1, seq_len, seq_len]
            returns: tuple (output tensor, attention weights)
        '''
from torch import nn
import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

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

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, v)
        self.attention_weights = attn_probs

        return output, attn_probs 
    
    def forward(self, x, mask=None, k=None, v=None):
        '''
            x, mask: torch.Tensor [batch_size, seq_len, d_model]
            returns: torch.Tensor
        '''

        batch_size, seq_len = x.size(0), x.size(1)

        k = k if k is not None else x
        v = v if v is not None else x

        k_seq_len = k.size(1)
        v_seq_len = v.size(1)

        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        k = self.w_k(k).view(batch_size, k_seq_len, self.n_heads, self.d_k).transpose(1,2)
        v = self.w_v(v).view(batch_size, v_seq_len, self.n_heads, self.d_k).transpose(1,2)

        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.w_o(attn_output)

        return output
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w2(self.dropout(self.activation(self.w1(x))))
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x+ self.dropout1(attn_output))

        ff_output=self.feed_forward(x)
        x = self.norm2(x+self.dropout2(ff_output))

        return x
    
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask = None, tgt_mask = None):
        self_attn = self.self_attention(x, tgt_mask)
        x = self.norm1(x+self.dropout1(self_attn))
        cross_attn = self.cross_attention(x=x, mask = src_mask, k=enc_output, v=enc_output)
        x = self.norm2(x+self.dropout2(cross_attn))
        ff_output = self.feed_forward(x)
        x = self.norm3(x+self.dropout3(ff_output))
        return x

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model = 512,
            n_heads = 8,
            n_layers = 6,
            d_ff = 2048,
            max_len = 512,
            dropout = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for block in self.encoder_blocks:
            x = block(x, mask)
        
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model = 512,
            n_heads = 8,
            n_layers = 6,
            d_ff = 2048,
            max_len = 512,
            dropout = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.output_projection.weight = self.token_embedding.weight

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for block in self.decoder_blocks:
            x = block(x, enc_output, src_mask, tgt_mask)
        
        return self.output_projection(x)
    

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        enc_output = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return out
    
    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return ~mask
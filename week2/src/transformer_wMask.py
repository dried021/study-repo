from torch import nn
from transformers import BertModel, BartModel
import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# transformer decoder only 모델 참고
# https://medium.com/@nibniw/building-a-large-language-model-from-scratch-a-comprehensive-technical-guide-eb2f4478663c
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # 5000 x 256
        pe = torch.zeros(max_len, d_model) #positional encoding

        # [0, 1, 2, .. , 4999]를 5000x1 크기로 변환
        # unsqeeze(dim) 차원 dim에 1인 차원을 생성
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 논문의 수식
        # pos / 10000 ^ (2i/d_model)
        # = pos * e^(-2i * ln(10000) / d_model)
        # 0부터 d_model까지 2간격으로 정수 생성
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        # 짝수 index에 sin 함수 적용
        pe[:, 0::2] = torch.sin(position * div_term)

        # 홀수 index에 cos 함수 적용
        pe[:, 1::2] = torch.cos(position * div_term)

        # 5000, 256 -> 1, 5000, 256
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # register_buffer(버터의 이름, 저장할 텐서)
        # pytorch의 텐서여도 parameter가 아니면 gpu로 이동이 되지 않음. -> buffer는 gpu에서 작동
        #   -> model.buffers()로 확인 가능
        # register_buffer를 사용하면 state_dict()에서 가중치 뿐만 아니라 이 값도 포함됨


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encodings added
        """
        # x.size는 입력의 길이
        # 입력의 길이에 맞춘 positional encodding을 더해줌
        return x + self.pe[:, :x.size(1)]
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d model must be divisible by n_heads"
        # 가정 설명문 (assert)
        # 뒤의 조건이 True가 아니면 AssertError를 발생시킴

        self.d_model = d_model
        self.n_heads = n_heads

        # 각각 head의 차원
        self.d_k = d_model // n_heads
        
        # Q K V를 만드는 가중치 행렬 256 -> 256 변환
        self.w_q = nn.Linear(d_model, d_model, bias = False)
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)
        self.w_o = nn.Linear(d_model, d_model, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def scaled_dot_product_attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes scaled dot-product attention with optional masking.
        
        Args:
            q: Query tensor, shape [batch_size, n_heads, seq_len, d_k]
            k: Key tensor, shape [batch_size, n_heads, seq_len, d_k]
            v: Value tensor, shape [batch_size, n_heads, seq_len, d_k]
            mask: Optional mask tensor, shape [batch_size, 1, seq_len, seq_len]
        Returns:
            Tuple of (output tensor, attention weights)
        """
        # Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V 구현

        # k.transpose: [batch_size, n_heads, seq_len, d_k] -> [batch_size, n_heads, d_k, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # batch_size, n_heads, seq_len, seq_len

        # 마스킹이 있을 경우 softmax에서 0이 나오게 하도록 하기 위해 -무한대로
        # mask가 0인 곳을 -1e9로 변환
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim= -1)
        attn_probs = self.dropout(attn_probs)

        # batch_size, n_heads, seq_len, seq_len @ batch_size, n_heads, seq_len, d_k => batch_size, n_heads, seq_len, d_k
        output = torch.matmul(attn_probs, v)

        self.attention_weights = attn_probs
        return output, attn_probs
    
    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Input tensor, shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor, shape [batch_size, seq_len, seq_len]
        Returns:
            Output tensor, shape [batch_size, seq_len, d_model]
        """
        
        batch_size, seq_len = x.size(0), x.size(1)

        # self.w_q(x): [batch_size, seq_len, d_model] -> d_model을 n_heads / d_k로 분할 후 seq_len과 n_heads의 순서를 바꿈
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)

        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # batch_size, n_heads, seq_len, d_k -> batch_size, seq_len, n_heads, d_k
        # view: n_heads와 d_k를 다시 d_model로 합침
        '''
        .contiguous()
        
        텐서의 shape를 조작하는 과정에서 메모리 저장 상태가 변경되는 경우가 있음
        메모리에 저장하는 자료 저장 순서가 원래의 방향과 어긋난 경우를 contiguous = False 상태라고 함
        보통 narrow(), view(), expand(), transpose() 메소드를 사용할 때 이 상태가 깨지게 됨

        view를 사용하기 위해 contiguous 상태여야 하므로 contiguous로 만들어 줌
        비연속적인 텐서를 연속적으로 만들어주는 역할

        '''
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.w_o(attn_output)
        
        # [batch_size, n_heads, d_model]
        return output
    
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network with GELU activation.

    최초 논문에서는 relu 사용
    position이 달라고 layer가 같으면 같은 weight를 사용하기 때문에 병렬 처리가 가능해짐
    """

    def __init__(self, d_model:int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation  = nn.GELU()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.activation(self.w1(x))))
    

    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout:float=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor, shape [batch_size, seq_len, seq_len]
        Returns:
            Output tensor
        """

        # Post-LN : norm(x+sublayer(x))
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            vocab_size : int,
            d_model: int = 512, 
            n_heads: int = 8,
            n_layers: int = 6,
            d_ff: int=2048,
            max_len: int = 512,
            dropout: float=0.1
    ):
        super().__init__()
        self.d_model = d_model

        # integer를 embedding vector로 변환하는 과정
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # layernorm weight을 0으로 초기화 하는 문제 
        # --> pytorch에서 자체적으로 초기화해주는 걸로 변경
        # self._init_parameters()

    # def _init_parameters(self):
    #     """
    #     Initialize parameters with Xavier uniform for weights and zero bias.
    #     """
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
    #         else:
    #             nn.init.zeros_(p)

    def forward(self, x:torch.Tensor, mask:Optional[torch.Tensor] = None ) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
            mask: Optional mask tensor, shape [batch_size, seq_len, seq_len]
        Returns:
            Encoded hidden states, shape [batch_size, seq_len, d_model]
        """
        # 논문에서도 sqrt(d_model)로 정규화한다고 되어 있음
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for encoder_blocks in self.encoder_blocks:
            if mask is not None and mask.dim() == 3:
                # [batch, 1, seq_len] -> [batch, 1, 1, seq_len]
                mask = mask.unsqueeze(2)

            x = encoder_blocks(x,mask)

        return x
        
    

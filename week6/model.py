import torch
import torch.nn as nn 
import math

class InputEmbedding(nn.Module):
    """ Create an instance for input embedding component.

    This layer transform the input token to the corresponding embedding vector of size (d_model x 1).

    Attributes:
        d_model: An Integer representing the output dimension of the embedding layer.
        vocab_size: An Integer representing the size of vocabulary.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """ Initialize the input embedding layer.

        Args:
            d_model: An Integer representing the output dimension of the embedding layer.
            vocab_size: An Integer representing the size of vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """ Forward function for input embedding layer.

        This function will transform the input token sequence x to the corresponding embedding vector of size (d_model x 1).

        Args:
            x: A tensor representing the input token sequence

        Returns:
            A tensor representing the embedding vector
        """
        # positional encoding과 크기를 맞추기 위해 root d_model을 곱함
        return self.embedding(x) * math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    """ Create an instance for positional encoding component.

    This layer add positional encoding informatiobn into the embedding vector.

    Attributes:
        d_model: An Integer representing the dimension of the model.
        seq_len: An Integer representing the length of the input sequence.
        dropout: A Float representing the dropout rate. 
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """ Initialize the positional encoding layer.

        Args:
            d_model: An Integer representing the dimension of the model.
            seq_len: An Integer representing the length of the input sequence.
            dropout: A Float representing the dropout rate.
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # initialize matrix of size (seq_len X d_model)
        pe = torch.zeros(seq_len, d_model)

        # [0, 1, 2, .. , 4999]를 5000x1 크기로 변환
        # unsqeeze(dim) 차원 dim에 1인 차원을 생성
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # pos / 10000 ^ (2i/d_model)
        # = pos * e^(-2i * ln(10000) / d_model)
        # 0부터 d_model까지 2간격으로 정수 생성
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        # apply sin to even and cos to odd position
        # 짝수에 sin 홀수에 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [seq_len, d_model] -> [1, seq_len, d_model]
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
        # register_buffer(버터의 이름, 저장할 텐서)
        # pytorch의 텐서여도 parameter가 아니면 gpu로 이동이 되지 않음. -> buffer는 gpu에서 작동
        #   -> model.buffers()로 확인 가능
        # register_buffer를 사용하면 state_dict()에서 가중치 뿐만 아니라 이 값도 포함됨

    def forward(self, x):
        """ Forward function for positional encoding layer.

        This function will add positional encoding information into the embedding vector.

        Args:
            x: A tensor representing the embedding vector.

        Returns:
            A tensor representing the output of the positional encoding layer.
        """

        # positional encoding은 fix된 값이기 때문에 gradient를 반영하지 않음
        # 미리 1, 최대 길이, 차원으로 맞춰둔 길이를 최대 길이가 아닌 실제 input의 길이에 맞춤
        # register buffer에 넣어서 계산이 안 되기는 하지만 가독성을 위한 표시
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)

        # 논문에서 명시한 것처럼 positional encoding과 embedding을 더한 후 dropout 실행
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """ Create an instance for layer normalization component.

    This layer performe the layer normalization on the input.

    Attributes:
        epsilon: A Float representing the epsilon value.
        alpha: A Float representing the alpha value (Multiplicative).
        bias: A Float representing the bias value (Additive).
    """
    def __init__(self, epsilon: float = 10**-6) -> None:
        """ Initialize the layer normalization layer.

        Args:
            epsilon: A Float representing the epsilon value. If not provided, the default value is 10**-6.
        """
        super().__init__()
        self.epsilon = epsilon

        # 각각 scalar 값
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # Additive

    def forward(self, x):
        """ Forward function for layer normalization layer.

        This function will normalize the input tensor according to the statistics of the input tensor.

        Args:
            x: A tensor representing the input.

        Returns:
            A tensor representing the normalized tensor.
        """
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)

        # nn.LayerNorm과 다른 점
        '''
        얘를 들어 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        이렇게 사용할 경우에는 
        γ and β are learnable affine transform parameters of normalized_shape if elementwise_affine is True 
        으로 되어 있음 elementwise_affine은 true인데 보통의 경우에 feature 개수의 차원을 가짐

        이 구현의 경우에는 alpha와 bias가 scalar이므로 feature 차원과 관계없이 정규화 후 동일하게 조정됨
        '''
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias
    
class FeedForwardBlock(nn.Module):
    """ Create an instance for feed forward block component.

    This layer pass the input through two linear layers performing the affine transformation on the input data.

    Attributes:
        d_model: An Integer representing the dimension of the model.
        d_ff: An Integer representing the dimension of the feed forward layer.
        dropout: A Float representing the dropout rate.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """ Initialize the feed forward block layer.

        Args:
            d_model: An Integer representing the dimension of the model.
            d_ff: An Integer representing the dimension of the feed forward layer.
            dropout: A Float representing the dropout rate.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x):
        """ Forward function for feed forward block layer.

        This function will pass the input through two linear layers performing the affine transformation on the input data.

        Args:
            x: A tensor representing the input.

        Returns:
            A tensor representing the output of the feed forward block layer.
        """
        x = self.linear1(x) # (batch, seq_len, d_model) --> (batch, seq_len, d_ff)

        # Relu 사용 nn.GELU()를 사용하기도 함
        x = torch.relu(x)  
        x = self.dropout(x)  
        out = self.linear2(x) # (batch, seq_len, d_ff) --> (batch, seq_len, d_model) 
        return out

class MultiHeadAttentionBlock(nn.Module):
    """ Create an instance for multi head attention block component.
    
    This layer perform the multi head attention on the input data.

    Attributes:
        d_model: An Integer representing the dimension of the model..
        num_head: An Integer representing the number of heads of the multi head attention.
        dropout: A Float representing the dropout rate.
    """
    def __init__(self, d_model: int, num_head: int, dropout: float) -> None:
        """ Initialize the multi head attention block layer.

        Args:
            d_model: An Integer representing the dimension of the model.
            num_head: An Integer representing the number of heads of the multi head attention.
            dropout: A Float representing the dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        assert d_model % num_head == 0, 'd_model must be divisible by num_head'
        # 가정 설명문 (assert)
        # 뒤의 조건이 True가 아니면 AssertError를 발생시킴

        # 각 head의 차원 --> d_k
        self.d_k = d_model // num_head

        # bias가 있는 linear 
        # q, k, v는 256->256 변환
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """ Compute the attention weights and output given the query, key, value and mask.

        Args:
            query: A tensor representing the query. 
            key: A tensor representing the key. 
            value: A tensor representing the value. 
            mask: A tensor representing the mask. 
            dropout: A Float representing the dropout rate.

        Returns:
            out: A tensor representing the output of multi-head attention layer. 
            attention_score: A tensor representing the attention score.
        """
        # query는 [batch, num_head, seq_len, d_k]
        # self.d_k를 왜 사용 안 하는지?
        d_k = query.shape[-1]

        # [batch, num_head, seq_len, d_k] --> [batch, num_head, seq_len, seq_len]
        # @ == torch.matmul(,)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # mask가 있다면 attention score를 -inf로 설정 (softmax에서 0이 나오도록)
        if mask is not None:
            # causal한 mask 뿐만 아니라 src_mask에서도 빈 값이 있는 공간에 -inf를 부여
            attention_score.masked_fill_(mask == 0, -1e9)

        attention_score = attention_score.softmax(dim=-1) # (batch, num_head, seq_len, seq_len)
        # .softmax = F.softmax()

        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score
    
    def forward(self, q, k, v, mask):
        """Forward function for multi-head attention layer.

        This function will compute the attention score and output given the query, key, value and mask.

        Args:
            q: A tensor representing the query. 
            k: A tensor representing the key. 
            v: A tensor representing the value. 
            mask: A tensor representing the mask. 

        Returns:
            x: A tensor representing the output of multi-head attention layer. 
            attention_score: A tensor representing the attention score.
        """
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, num_head, d_k) --> (batch, num_head, seq_len, d_k)
        # d_model을 head의 차원으로 분할 후 순서를 바꿈
        query = query.view(query.shape[0], query.shape[1], self.num_head, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_head, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_head, self.d_k).transpose(1,2)

        # 후의 visualization을 위한 저장..
        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, num_head, seq_len, d_k) --> (batch, seq_len, num_head, d_k) --> (batch, seq_len, d_model)
        # 0번쨰 차원은 batch, num_head * d_k는 d_model
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.num_head*self.d_k)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    """ Create an instance for residual connection component.

    This function will perform the residual connection connecting the input and output of the sublayer to the following layer.

    Attributes:
        dropout: A Float representing the dropout rate.
        norm: A LayerNormalization layer.
    """
    def __init__(self, dropout: float) -> None:
        """ Initialize the residual connection layer.

        Args:
            dropout: A Float representing the dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """ Forward function for residual connection layer.

        This function will perform the residual connection connecting the input and output of the sublayer to the following layer.

        Args:
            x: A tensor representing the input.
            sublayer: A callable representing the sublayer.

        Returns:
            A tensor representing the output of the residual connection layer.
        """

        # normalization을 하고 넣은 sublayer를 거쳐 dropout 후 residual connection하는 것까지 구현
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    """ Create an instance for encoder block component.

    Create a encoder block with self attention and feed forward block.

    Attributes:
        self_attention_block: A MultiHeadAttentionBlock layer.
        feed_forward_block: A FeedForwardBlock layer.
        residual_connection: A ResidualConnection layer.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """ Initialize the encoder block layer.

        Args:
            self_attention_block: A MultiHeadAttentionBlock layer.
            feed_forward_block: A FeedForwardBlock layer.
            dropout: A Float representing the dropout rate.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        # transformer에서 attention block 한 번 feed_forward한 번 sublayer 후 
        # normalization + dropout + residual connection을 하게 됨 
        # 이를 modulelist로 묶어 수행할 수 있도록 함
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """ Forward function for encoder block layer.

        This function will pass the input through multi-head attention, feed forward block and perform the residual connection.

        Args:
            x: A tensor representing the input.
            src_mask: A tensor representing the source mask.

        Returns:
            A tensor representing the output of the encoder block layer.
        """
        # encoder에서는 query, key, value가 모두 input값이 동일
        # src_mask는 input seq가 얼마나 차지하는 지를 알려줌
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    """ Create an instance for encoder component.

    Create a encoder with multiple encoder block.

    Attributes:
        layers: A ModuleList of EncoderBlock layers.
        norm: A LayerNormalization layer.
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        """ Initialize the encoder layer.

        Args:
            layers: A ModuleList of EncoderBlock layers.
        """
        super().__init__()
        self.layers = layers
        self.norm  = LayerNormalization()

    def forward(self, x, mask):
        """ Forward function for encoder layer.

        This function will pass the input through multiple encoder block and perform the layer normalization.

        Args:
            x: A tensor representing the input.
            mask: A tensor representing the source mask.

        Returns:
            A tensor representing the output of the encoder layer.
        """

        # modulelist를 실행시킴
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    """ Create an instance for decoder block component.

    Create a decoder block with self attention, cross attention, feed forward block, and the residual connection.

    Attributes:
        self_attention: A MultiHeadAttentionBlock layer.
        cross_attention: A MultiHeadAttentionBlock layer.
        feed_forward_block: A FeedForwardBlock layer.
        residual_connection: A ResidualConnection layer.
    """
    def __init__(self, self_attention: MultiHeadAttentionBlock, cross_attention: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """ Initialize the decoder block layer.

        Args:
            self_attention: A MultiHeadAttentionBlock layer.
            cross_attention: A MultiHeadAttentionBlock layer.
            feed_forward_block: A FeedForwardBlock layer.
            dropout: A Float representing the dropout rate.
        """
        super().__init__()

        # 두 가지의 attention을 각각 정의
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward_block = feed_forward_block
        
        # encoder와 마찬가지로 modulelist로 residual connection + normalization + dropout을 정의
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """ Forward function for decoder block layer.

        This function will pass the input through multiple attention and feed forward block, and perform the residual connection.

        Args:
            x: A tensor representing the input.
            encoder_output: A tensor representing the output of the encoder layer.
            src_mask: A tensor representing the source mask.
            tgt_mask: A tensor representing the target mask.

        Returns:
            A tensor representing the output of the decoder block layer.
        """
        # masked self attention (tgt_mask에 미래 정보 masking도 포함)
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))

        # encoder cross attention
        # query는 그 encoder의 마지막 출력, key & query는 그 전 레이어 decoder의 출력
        # value가 encoder의 output이므로 source mask 사용
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))

        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    """ Create an instance for decoder component.

    Create a decoder with multiple decoder block.

    Attributes:
        layers: A ModuleList of DecoderBlock layers.
        norm: A LayerNormalization layer.
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        """ Initialize the decoder layer.

        Args:
            layers: A ModuleList of DecoderBlock layers.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """ Forward function for decoder layer.

        This function will pass the input through multiple decoder block, and perform the layer normalization.

        Args:
            x: A tensor representing the input.
            encoder_output: A tensor representing the output of the encoder layer.
            src_mask: A tensor representing the source mask.
            tgt_mask: A tensor representing the target mask.

        Returns:
            A tensor representing the output of the decoder layer.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """ Create an instance for projection layer component.

    Create a projection layer with log softmax and linear layer to project the output of the decoder to the vocabulary.

    Attributes:
        projection: A Linear layer.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """ Initialize the projection layer.

        Args:
            d_model: An Integer representing the dimension of the model.
            vocab_size: An Integer representing the size of vocabulary.
        """
        super().__init__()

        # vector -> 단어 해주는 layer
        self.projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """ Forward function for projection layer.

        This function will project the output of the decoder to the vocabulary size and apply the log softmax.

        Args:
            x: A tensor representing the output of the decoder.

        Returns:
            A tensor representing the output of the projection layer.
        """
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        # 단어 크기로 dimension projection 후 softmax 적용
        return torch.log_softmax(self.projection(x), dim=-1)
    
class Transformer(nn.Module):
    """ Create an instance for transformer model.

    Create a transformer model with encoder, decoder, source embedding, target embedding, source positional encoding, target positional encoding, and projection layer.

    Attributes:
        encoder: A Encoder layer.
        decoder: A Decoder layer.
        src_embed: A InputEmbedding layer.
        tgt_embed: A InputEmbedding layer.
        src_pos: A PositionalEncoding layer.
        tgt_pos: A PositionalEncoding layer.
        projection_layer: A ProjectionLayer layer.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        """ Initialize the transformer model.

        Args:
            encoder: A Encoder layer.
            decoder: A Decoder layer.
            src_embed: A InputEmbedding layer.
            tgt_embed: A InputEmbedding layer.
            src_pos: A PositionalEncoding layer.
            tgt_pos: A PositionalEncoding layer.
            projection_layer: A ProjectionLayer layer.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """ Encode the source sequence.

        This function will embed the source sequence, add the positional encoding, and feed it into the encoder.

        Args:
            src: A tensor representing the source sequence.
            src_mask: A tensor representing the source mask.

        Returns:
            A tensor representing the output of the encoder.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """ Decode the target sequence.

        This function will embed the target sequence, add the positional encoding, and feed it into the decoder.

        Args:
            encoder_output: A tensor representing the output of the encoder.
            src_mask: A tensor representing the source mask.
            tgt: A tensor representing the target sequence.
            tgt_mask: A tensor representing the target mask.

        Returns:
            A tensor representing the output of the decoder.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        """ Project the output of the decoder to the target vocabulary.

        This function will apply the projection layer to the output of the decoder, and return the result.

        Args:
            x: A tensor representing the output of the decoder.

        Returns:
            A tensor representing the output of the projection layer.
        """
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """Build a transformer model.

    Args:
        src_vocab_size (int): The number of unique words in the source language.
        tgt_vocab_size (int): The number of unique words in the target language.
        src_seq_len (int): The length of the sequence in the source language.
        tgt_seq_len (int): The length of the sequence in the target language.
        d_model (int, optional): The dimensionality of the model. Defaults to 512.
        N (int, optional): The number of encoder and decoder layers. Defaults to 6.
        h (int, optional): The number of heads in the multi-head attention. Defaults to 8.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        d_ff (int, optional): The dimensionality of the feed forward layer. Defaults to 2048.

    Returns:
        Transformer: The transformer model.
    """
    # Create embedding layers
    # src와 tgt 따로 embedding
    # 원논문에서는 source와 target을 같이 넣은 vocab을 사용해서 하나의 weight으로 가능했음
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create positional encoding layers
    # 각각 positional Encoding
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create Encoder blocks
    # 레이어 개수 만큼 각 sublayer의 요소들을 생성
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters
    # Xavier로 parameter 초기화
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
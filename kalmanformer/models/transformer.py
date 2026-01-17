import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional Encoding as described in the prompt:
    P(k,2i) = sin(k / n^(2i/d))
    P(k,2i+1) = cos(k / n^(2i/d))
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term calculation: n^(2i/d) term
        # exp(2i * -log(n)/d)
        # Using 10000 as base 'n' is standard, prompt says 'n' but usually implies the standard base
        # If 'n' is state dim, that would be weird. Standard is 10000.
        # "P(k, 2i) = sin(k / n^(2i/d))" -> The prompt likely refers to the standard n=10000.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # Add PE to input
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention custom implementation.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Attention(Q,K,V) = softmax(Q*K^T / sqrt(d_k)) * V
        """
        batch_size = Q.size(0)
        
        # Scores: (batch, heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear Projections and Split Heads
        # (batch, seq, d_model) -> (batch, seq, heads, d_k) -> (batch, heads, seq, d_k)
        Q = self.W_Q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concat Heads: (batch, heads, seq, d_k) -> (batch, seq, heads, d_k) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final Linear
        return self.W_O(attn_output)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Sublayer 1: Self Attention
        # Residual connection
        residual = x
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = self.norm1(residual + x)
        
        # Sublayer 2: Feed Forward
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.norm2(residual + x)
        
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # Sublayer 1: Self Attention
        residual = x
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = self.norm1(residual + x)
        
        # Sublayer 2: Cross Attention (Query=x, Key/Value=memory)
        residual = x
        x = self.cross_attn(x, memory, memory, memory_mask)
        x = self.dropout(x)
        x = self.norm2(residual + x)
        
        # Sublayer 3: Feed Forward
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.norm3(residual + x)
        
        return x

class TransformerModel(nn.Module):
    def __init__(self, 
                 input_dim_enc: int, 
                 input_dim_dec: int, 
                 d_model: int = 32, 
                 num_heads: int = 2, 
                 num_layers: int = 2, 
                 d_ff: int = 64, 
                 dropout: float = 0.1):
        super().__init__()
        
        # Input Projections -> d_model
        self.enc_embedding = nn.Linear(input_dim_enc, d_model)
        self.dec_embedding = nn.Linear(input_dim_dec, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.d_model = d_model

    def forward(self, src, tgt):
        # src shape: [batch, seq_len, input_dim_enc]
        # tgt shape: [batch, seq_len, input_dim_dec]
        
        # Embed and Add Position
        src_emb = self.enc_embedding(src)
        src_emb = self.pos_encoder(src_emb)
        
        tgt_emb = self.dec_embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Encoder
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory)
            
        # Decoder
        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, memory)
            
        return output

import torch
import torch.nn as nn
import math

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Simple Transformer Encoder to understand sentence structure
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, text_indices, mask=None):
        # text_indices: (Batch, SeqLen)
        
        # 1. Get embeddings first
        x = self.embedding(text_indices)
        
        # 2. Scale embeddings
        x = x * math.sqrt(x.shape[-1])
        
        # 3. Add positional encoding
        x = self.pos_encoder(x)
        
        # 4. Pass through Transformer
        # Output: (Batch, SeqLen, EmbedDim) -> e.g. (B, 15, 256)
        context = self.transformer(x, src_key_padding_mask=mask)
        return context

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
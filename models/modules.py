import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        assert hidden_dim % 2 == 0, "Hidden dimension for TimeEmbedding must be even."
        self.hidden_dim = hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t):
        # t: (batch_size,)
        device = t.device
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.hidden_dim, 2, device=device).float() / self.hidden_dim))
        sinusoid_inp = torch.outer(t, inv_freq)
        embedding = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return self.mlp(embedding)


class Block(nn.Module):
    """ResNet block with GroupNorm, SiLU, and Time Embedding injection."""
    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.silu = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        if in_channels == out_channels:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, time_emb):
        h = self.silu(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding (broadcast over spatial dims)
        time_emb_proj = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_emb_proj

        h = self.silu(self.norm2(h))
        h = self.conv2(h)

        return h + self.residual_conv(x)


class CrossAttention(nn.Module):
    """Spatial Cross Attention module."""
    def __init__(self, query_dim, context_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.norm = nn.LayerNorm(query_dim)
        self.query_projection = nn.Linear(query_dim, query_dim)
        self.key_projection = nn.Linear(context_dim, query_dim)
        self.value_projection = nn.Linear(context_dim, query_dim)

        self.mha = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(query_dim)

    def forward(self, x, context):
        # x: (B, C, H, W) -> Treated as Sequence
        # context: (B, Window, Markers, C_ctx) -> Flattened to Sequence
        
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        x_norm = self.norm(x_flat)

        q = self.query_projection(x_norm)
        
        # Flatten context: (B, T, M, C) -> (B, T*M, C)
        # CORRECTION: Utiliser .reshape() au lieu de .view()
        # Le tenseur 'context' provient de l'encodeur qui utilise permute, 
        # donc il n'est pas contigu.
        context_flat = context.reshape(b, -1, self.context_dim) 
        k = self.key_projection(context_flat)
        v = self.value_projection(context_flat)

        attn_out, _ = self.mha(q, k, v)
        
        x_attn = x_flat + self.dropout(attn_out)
        x_attn = self.norm2(x_attn)
        x_ffn = x_attn + self.dropout(self.ffn(x_attn))

        # Reshape back to image format
        return x_ffn.permute(0, 2, 1).view(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CrossAttention1D(nn.Module):
    def __init__(self, channels, context_dim):
        super().__init__()
        self.q_proj = nn.Conv1d(channels, channels, 1)
        self.k_proj = nn.Linear(context_dim, channels)
        self.v_proj = nn.Linear(context_dim, channels)
        self.scale = channels ** -0.5

    def forward(self, x, context):
        # x: (B, C, T)  |  context: (B, SeqLen, C_ctx)
        q = self.q_proj(x).permute(0, 2, 1) # (B, T, C)
        k = self.k_proj(context)            # (B, Seq, C)
        v = self.v_proj(context)            # (B, Seq, C)
        
        # Match text context to motion frames
        attn = torch.einsum('btc, bsc -> bts', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bts, bsc -> btc', attn, v)
        out = out.permute(0, 2, 1) # (B, C, T)
        return x + out

class DiffusionBlock1D(nn.Module):
    def __init__(self, channels, context_dim, time_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_dim, channels)
        self.attn = CrossAttention1D(channels, context_dim)
        self.act = nn.SiLU()

    def forward(self, x, t_emb, context):
        res = x
        h = self.act(self.conv1(x))
        h = h + self.time_mlp(t_emb).unsqueeze(-1) # Broadcast time over frames
        h = self.act(self.conv2(h))
        h = self.attn(h, context) # Look at the text words
        return h + res

class LatentDiffusion1D(nn.Module):
    """A specialized 1D Diffusion model for VQ-VAE Latents"""
    def __init__(self, in_channels=256, model_channels=256, context_dim=256, num_blocks=4):
        super().__init__()
        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.init_conv = nn.Conv1d(in_channels, model_channels, 1)
        
        self.blocks = nn.ModuleList([
            DiffusionBlock1D(model_channels, context_dim, time_dim) 
            for _ in range(num_blocks)
        ])
        
        self.final_conv = nn.Conv1d(model_channels, in_channels, 1)

    def forward(self, x, t, context):
        # x: (B, 256, 15)
        t_emb = self.time_mlp(t)
        
        x = self.init_conv(x)
        for block in self.blocks:
            x = block(x, t_emb, context)
            
        return self.final_conv(x)
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import TimeEmbedding, Block, Downsample, Upsample, CrossAttention

class ConditionalDiffusionModel(nn.Module):
    """
    Robust UNet implementation that supports dynamic depths and handling
    of odd spatial dimensions via interpolation.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        context_dim=256,
        time_emb_dim=256,
        ch_mults=(1, 2, 4),        # Depth multipliers
        is_attention=(False, True, True), # Attention at each resolution?
        num_res_blocks=2,
        num_heads=8,
        num_groups=8
    ):
        super().__init__()

        self.model_channels = model_channels
        self.time_embed = TimeEmbedding(time_emb_dim)
        
        # --- Downsampling Path ---
        self.input_blocks = nn.ModuleList()
        
        # Initial Conv
        self.input_blocks.append(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        
        input_block_chans = [model_channels] # Track channels for skip connections
        ch = model_channels
        
        for level, mult in enumerate(ch_mults):
            out_ch = model_channels * mult
            
            # Add Residual Blocks
            for _ in range(num_res_blocks):
                layers = [Block(ch, out_ch, time_emb_dim, num_groups)]
                ch = out_ch
                
                # Add Attention if enabled for this level
                if is_attention[level]:
                    layers.append(CrossAttention(ch, context_dim, num_heads))
                
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)

            # Add Downsample (except at last level)
            if level != len(ch_mults) - 1:
                self.input_blocks.append(Downsample(ch))
                input_block_chans.append(ch)

        # --- Middle Block (Bottleneck) ---
        self.middle_block = nn.Sequential(
            Block(ch, ch, time_emb_dim, num_groups),
            CrossAttention(ch, context_dim, num_heads),
            Block(ch, ch, time_emb_dim, num_groups)
        )

        # --- Upsampling Path ---
        self.output_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(ch_mults))):
            out_ch = model_channels * mult
            
            # We add +1 block because we process the skip connection
            for i in range(num_res_blocks + 1):
                skip_ch = input_block_chans.pop()
                
                layers = [Block(ch + skip_ch, out_ch, time_emb_dim, num_groups)]
                ch = out_ch
                
                if is_attention[level]:
                    layers.append(CrossAttention(ch, context_dim, num_heads))
                
                self.output_blocks.append(nn.Sequential(*layers))
                
            # Upsample (except at first level)
            if level != 0:
                self.output_blocks.append(Upsample(ch))

        # --- Final Output ---
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, context):
        # x: (Batch, Window, Markers, 3)
        b, w, m, _ = x.shape
        
        # 1. Permute to Image Format: (B, 3, W, M)
        x = x.permute(0, 3, 1, 2)
        
        # 2. Time Embedding
        t_emb = self.time_embed(t)

        # 3. Downsampling
        hs = [] # Store skip connections
        
        # Process initial conv (first item in input_blocks is just a Conv2d, not Sequential with time)
        h = self.input_blocks[0](x)
        hs.append(h)
        
        # Process remaining blocks
        for module in self.input_blocks[1:]:
            if isinstance(module, Downsample):
                h = module(h)
            else:
                # It's a Sequential of [Block, optional Attention]
                # We need to manually inject time_emb into the Block
                for layer in module:
                    if isinstance(layer, Block):
                        h = layer(h, t_emb)
                    elif isinstance(layer, CrossAttention):
                        h = layer(h, context)
            hs.append(h)

        # 4. Middle
        for layer in self.middle_block:
            if isinstance(layer, Block):
                h = layer(h, t_emb)
            elif isinstance(layer, CrossAttention):
                h = layer(h, context)

        # 5. Upsampling
        for module in self.output_blocks:
            if isinstance(module, Upsample):
                h = module(h)
            else:
                # Retrieve skip connection
                h_skip = hs.pop()
                
                # Spatial mismatch fix (your robust addition)
                if h.shape[2:] != h_skip.shape[2:]:
                    h = F.interpolate(h, size=h_skip.shape[2:], mode='nearest')
                
                # Concatenate
                h = torch.cat([h, h_skip], dim=1)
                
                # Pass through Block/Attn
                for layer in module:
                    if isinstance(layer, Block):
                        h = layer(h, t_emb)
                    elif isinstance(layer, CrossAttention):
                        h = layer(h, context)

        # 6. Final Output
        out = self.out(h)
        
        # Ensure output matches original input spatial size
        if out.shape[2:] != (w, m):
            out = F.interpolate(out, size=(w, m), mode='bilinear', align_corners=False)
            
        # Permute back: (B, 3, W, M) -> (B, W, M, 3)
        return out.permute(0, 2, 3, 1)
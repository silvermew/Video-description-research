import torch
import torch.nn as nn

class MotionEncoder(nn.Module):
    """
    Encodes marker-based motion data.
    Input:  (Batch, Window, Markers, 3)
    Output: (Batch, Window, Markers, Hidden_Dim)
    """
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Project 3D coordinates to hidden dimension
        self.linear_projection = nn.Linear(3, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, w, m, _ = x.shape

        # (B, W, M, 3) -> (B, W, M, H)
        x = self.linear_projection(x)

        # Merge Batch and Markers to process temporal sequence: (B*M, W, H)
        x = x.permute(0, 2, 1, 3).reshape(b * m, w, self.hidden_dim)
        
        # Apply Transformer over time
        x = self.transformer(x)

        # Restore: (B, W, M, H)
        x = x.reshape(b, m, w, self.hidden_dim).permute(0, 2, 1, 3)
        return x
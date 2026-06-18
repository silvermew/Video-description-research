
import torch
import torch.nn as nn
from config import INPUT_FEATURE_DIM, CONTEXT_DIM

class CaptioningBridge(nn.Module):
    """
    Bridge class to handle 4D -> 3D shape mismatch for MotionCaptioner.
    Adapted from inference_captioning.py
    """
    def __init__(self, base_model, device):
        super().__init__()
        self.base_model = base_model
        self.device = device
        self.adapter = nn.Linear(INPUT_FEATURE_DIM, CONTEXT_DIM).to(device)
        self.embed_layer = getattr(base_model, 'text_embedding', getattr(base_model, 'embedding', None))
        self.pos_emb_param = getattr(base_model, 'text_pos_embedding', None)
        self.out_layer = getattr(base_model, 'fc_out', getattr(base_model, 'linear', None))

    def encode_motion(self, motions):
        if motions.dim() == 2: motions = motions.unsqueeze(0)
        
        # 3D Input (Batch, Time, 263) -> Adapter -> Transformer
        if motions.dim() == 3:
            x = self.adapter(motions)
            # Access the transformer inside the motion encoder
            x = self.base_model.motion_encoder.transformer(x)
            return x
            
        # 4D Input (Batch, Window, Markers, 3) -> Squash -> Transformer
        elif motions.dim() == 4:
            context = self.base_model.motion_encoder(motions)
            # Average over markers to get (Batch, Window, HiddenDim)
            return context.mean(dim=2)
            
        return None

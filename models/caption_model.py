import torch
import torch.nn as nn
# [CRITICAL FIX] Use relative import with a dot (.)
from .encoder import MotionEncoder 

# [CRITICAL FIX] Class name must be 'MotionCaptioner'
class MotionCaptioner(nn.Module):
    def __init__(self, 
                 motion_encoder, 
                 vocab_size, 
                 text_embed_dim=256, 
                 num_heads=8, 
                 num_layers=4):
        super().__init__()
        
        # 1. Motion Encoder (Pre-trained)
        self.motion_encoder = motion_encoder
        
        # Freeze the motion encoder
        for param in self.motion_encoder.parameters():
            param.requires_grad = False
            
        # 2. Text Embedding
        self.embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.text_pos_embedding = nn.Parameter(torch.randn(1, 50, text_embed_dim))
        
        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=text_embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 4. Final Head
        self.fc_out = nn.Linear(text_embed_dim, vocab_size)

    def forward(self, motion_window, text_input, tgt_mask=None):
        # A. Encode Motion
        with torch.no_grad():
            # This returns (Batch, Frames, Markers, HiddenDim) -> e.g. (8, 60, 64, 256)
            motion_context_4d = self.motion_encoder(motion_window)
            
            # [CRITICAL FIX] Collapse the 'Markers' dimension (Dim 2)
            # We average the 64 markers to get one "pose vector" per frame.
            # New shape: (Batch, Frames, HiddenDim) -> (8, 60, 256)
            motion_context = motion_context_4d.mean(dim=2) 
            
        # B. Embed Text
        B, SeqLen = text_input.shape
        text_emb = self.embedding(text_input)
        
        # Add Positional Encoding safely
        max_pos = self.text_pos_embedding.shape[1]
        # Slice the positional embedding to match the text length
        text_emb = text_emb + self.text_pos_embedding[:, :SeqLen, :]
        
        # C. Decode
        # Now 'memory' (motion_context) is 3D, so the Transformer will accept it.
        output = self.decoder(
            tgt=text_emb, 
            memory=motion_context, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=(text_input == 0)
        )
        
        # D. Predict Words
        prediction = self.fc_out(output)
        return prediction
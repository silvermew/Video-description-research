import torch
import torch.nn as nn
import torch.nn.functional as F

def perplexity(encoding_indices, num_embeddings):
    """Calculates how many codes are actually being used (higher is better)."""
    avg_probs = torch.argmax(encoding_indices, dim=1).float().histc(bins=num_embeddings, min=0, max=num_embeddings)
    avg_probs = avg_probs / torch.sum(avg_probs)
    return torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

class VectorQuantizer(nn.Module):
    """
    Discretizes the continuous latent vectors into codebook indices.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self._embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        return loss, quantized, perplexity(encoding_indices, self._num_embeddings)

# --- NEW: ResNet Block ---
class Resnet1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return x + self.block(x) # The Residual "Skip" Connection

class MotionVQVAE(nn.Module):
    def __init__(self, input_dim=192, latent_dim=256, num_tokens=512):
        super(MotionVQVAE, self).__init__()
        
        # UPGRADED ENCODER (Deeper, smarter)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=4, stride=2, padding=1), # 60 -> 30
            nn.ReLU(),
            Resnet1D(256),
            Resnet1D(256),
            nn.Conv1d(256, latent_dim, kernel_size=4, stride=2, padding=1), # 30 -> 15
            Resnet1D(latent_dim),
            Resnet1D(latent_dim),
        )
        
        self.pre_vq_conv = nn.Conv1d(latent_dim, latent_dim, kernel_size=1)
        self.vq = VectorQuantizer(num_tokens, latent_dim, commitment_cost=0.25)
        
        # UPGRADED DECODER (Deeper, smarter)
        self.decoder = nn.Sequential(
            Resnet1D(latent_dim),
            Resnet1D(latent_dim),
            nn.ConvTranspose1d(latent_dim, 256, kernel_size=4, stride=2, padding=1), # 15 -> 30
            nn.ReLU(),
            Resnet1D(256),
            Resnet1D(256),
            nn.ConvTranspose1d(256, input_dim, kernel_size=4, stride=2, padding=1), # 30 -> 60
        )

    def forward(self, x):
        b, t, m, c = x.shape
        x_flat = x.view(b, t, m*c).permute(0, 2, 1)
        
        z = self.encoder(x_flat)
        z = self.pre_vq_conv(z)
        
        vq_loss, quantized, perplexity_val = self.vq(z)
        
        x_recon = self.decoder(quantized)
        x_recon = x_recon.permute(0, 2, 1).view(b, t, m, c)
        
        return vq_loss, x_recon, perplexity_val
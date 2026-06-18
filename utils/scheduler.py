import torch

class NoiseScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.arange(
            0, 
            self.num_train_timesteps, 
            dtype=torch.long
        ).flip(0)
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, original_samples, noise, timesteps):
        # Move scheduler tensors to the same device as inputs
        device = original_samples.device
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)

        # Reshape for broadcasting
        batch_size = original_samples.shape[0]
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(batch_size, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(batch_size, 1, 1, 1)

        return sqrt_alpha * original_samples + sqrt_one_minus * noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.num_train_timesteps, (batch_size,))
    
    def step(self, model_output, timestep, sample):
    
        # Déplacer les tenseurs du scheduler sur le même appareil que les entrées
        device = sample.device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)

        # 1. Mise à l'échelle des paramètres (pour broadcasting)
        t = timestep # Timestep est un tenseur de taille (1,) ici
        alpha_t = self.alphas[t].view(-1, 1, 1, 1) # alpha_t
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1) # alpha_bar_t
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        
        # 2. Calculer le terme pour l'estimation X_0 (souvent appelé pred_x0)
        # pred_x0 = (X_t - sqrt(1 - alpha_bar_t) * predicted_noise) / sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
        
        # Estimer X_0 (la donnée nette)
        pred_x0 = (sample - sqrt_one_minus_alpha_cumprod_t * model_output) / torch.sqrt(alpha_cumprod_t)
        
        # 3. Calculer la moyenne de X_{t-1} (mu_t)
        # mu_t = (X_t - (beta_t / sqrt(1 - alpha_bar_t)) * predicted_noise) / sqrt(alpha_t)
        # Version en utilisant pred_x0:
        mu = (1.0 / torch.sqrt(alpha_t)) * (sample - (beta_t / sqrt_one_minus_alpha_cumprod_t) * model_output)
        
        # 4. Calculer le terme de variance et ajouter du bruit stochastique (DDPM)
        variance = beta_t
        noise = torch.randn_like(sample)
        
        # Terme stochastique (si t > 0)
        prev_sample = mu + torch.sqrt(variance) * noise
        
        # Si t=0, pas de bruit à ajouter
        if (t == 0).all():
            prev_sample = mu
            
        # --- Sortie ---
        class SchedulerOutput:
            def __init__(self, prev_sample):
                self.prev_sample = prev_sample

        return SchedulerOutput(prev_sample)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Configuration
T = 1000  # Number of diffusion steps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Noise schedule (linear beta)
def get_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# Precompute constants
betas = get_beta_schedule(T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


# Forward diffusion q(x_t | x_0)
def forward_diffusion_sample(x_0, t, noise=None):
    """
    Takes an image and a timestep t, and returns a noisy version of it
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t])[:, None]
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[t])[:, None]
    return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise, noise


# Simple MLP model
class DenoiseMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x, t):
        # t is timestep, reshape and concatenate
        t = t.float() / T
        t = t.unsqueeze(-1)
        x_in = torch.cat([x, t], dim=1)
        return self.net(x_in)


# Training loop (toy dataset: Gaussian blobs)
def train_diffusion():
    model = DenoiseMLP(input_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 2000
    batch_size = 128

    for epoch in range(epochs):
        x_0 = torch.randn(batch_size, 2).to(device)  # e.g., Gaussian blobs
        t = torch.randint(0, T, (batch_size,), device=device).long()
        x_noisy, noise = forward_diffusion_sample(x_0, t)
        noise_pred = model(x_noisy, t)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    return model


# Sampling (denoising process)
@torch.no_grad()
def sample(model, n_samples=100):
    x = torch.randn(n_samples, 2).to(device)
    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_hat = alphas_cumprod[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat)) * predicted_noise
        ) + torch.sqrt(beta_t) * noise

    return x.cpu()


# Run training and sample
if __name__ == "__main__":
    model = train_diffusion()
    samples = sample(model)

    # Visualize (optional)
    import matplotlib.pyplot as plt
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6)
    plt.title("Generated Samples")
    plt.axis('equal')
    plt.show()

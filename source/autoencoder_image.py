import math
import random

import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch
import torch.nn as nn
from pyro.nn import PyroModule
from skimage import color, data, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt


pyro.clear_param_store()
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a public sample image from scikit-image
image = img_as_float(data.astronaut())          # shape: (H, W, 3), values in [0, 1]
image = color.rgb2gray(image).astype(np.float32)  # shape: (H, W)

# Add synthetic corruption for image restoration
noise_std = 0.20
noisy_image = np.clip(image + noise_std * np.random.randn(*image.shape).astype(np.float32), 0.0, 1.0)

# Convert image into overlapping patches
def extract_patches(img: np.ndarray, patch_size: int = 16, stride: int = 8) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    h, w = img.shape
    patches = []
    positions = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patches.append(img[i:i + patch_size, j:j + patch_size].reshape(-1))
            positions.append((i, j))
    patches = np.stack(patches).astype(np.float32)
    return torch.from_numpy(patches), positions

def reconstruct_from_patches(
    patches: torch.Tensor,
    positions: list[tuple[int, int]],
    image_shape: tuple[int, int],
    patch_size: int = 16,
) -> np.ndarray:
    h, w = image_shape
    canvas = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    patches_np = patches.detach().cpu().numpy().reshape(-1, patch_size, patch_size)

    for patch, (i, j) in zip(patches_np, positions):
        canvas[i:i + patch_size, j:j + patch_size] += patch
        weight[i:i + patch_size, j:j + patch_size] += 1.0

    return np.divide(canvas, np.maximum(weight, 1e-8))


patch_size = 16
stride = 8
clean_patches, positions = extract_patches(image, patch_size=patch_size, stride=stride)
noisy_patches, _ = extract_patches(noisy_image, patch_size=patch_size, stride=stride)
clean_patches = clean_patches.to(device)
noisy_patches = noisy_patches.to(device)
input_dim = patch_size * patch_size
latent_dim = 64
hidden_dim = 256

# Deterministic autoencoder in Pyro: only likelihood, no latent prior
class DenoisingAutoencoder(PyroModule):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = PyroModule[nn.Sequential](
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = PyroModule[nn.Sequential](
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

model_net = DenoisingAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

def model(x_noisy: torch.Tensor, x_clean: torch.Tensor):
    pyro.module("autoencoder", model_net)
    x_hat = model_net(x_noisy)

    sigma = pyro.param(
        "sigma",
        lambda: torch.tensor(0.10, device=device),
        constraint=dist.constraints.positive,
    )

    with pyro.plate("data", x_clean.shape[0]):
        pyro.sample("obs", dist.Normal(x_hat, sigma).to_event(1), obs=x_clean)


def guide(x_noisy: torch.Tensor, x_clean: torch.Tensor):
    return None

optimizer = Adam({"lr": 1e-3})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Train
num_epochs = 600
loss_history = []

for epoch in range(num_epochs):
    loss = svi.step(noisy_patches, clean_patches)
    loss_history.append(loss / noisy_patches.shape[0])

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1:4d} | Loss per patch: {loss_history[-1]:.6f} | sigma={pyro.param('sigma').item():.4f}")

# Restore the full image from denoised patches
with torch.no_grad():
    restored_patches = model_net(noisy_patches)

restored_image = reconstruct_from_patches(
    restored_patches,
    positions,
    image_shape=image.shape,
    patch_size=patch_size,
)
restored_image = np.clip(restored_image, 0.0, 1.0)

# Evaluation
psnr_noisy = peak_signal_noise_ratio(image, noisy_image, data_range=1.0)
psnr_restored = peak_signal_noise_ratio(image, restored_image, data_range=1.0)

ssim_noisy = structural_similarity(image, noisy_image, data_range=1.0)
ssim_restored = structural_similarity(image, restored_image, data_range=1.0)

print(f"\nPSNR noisy     : {psnr_noisy:.4f}")
print(f"PSNR restored  : {psnr_restored:.4f}")
print(f"SSIM noisy     : {ssim_noisy:.4f}")
print(f"SSIM restored  : {ssim_restored:.4f}")

# Visualize
plt.figure(figsize=(15, 4))
plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")
plt.subplot(1, 4, 2)
plt.imshow(noisy_image, cmap="gray")
plt.title(f"Noisy\nPSNR={psnr_noisy:.2f}, SSIM={ssim_noisy:.3f}")
plt.axis("off")
plt.subplot(1, 4, 3)
plt.imshow(restored_image, cmap="gray")
plt.title(f"Restored\nPSNR={psnr_restored:.2f}, SSIM={ssim_restored:.3f}")
plt.axis("off")
plt.subplot(1, 4, 4)
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("ELBO loss / patch")
plt.tight_layout()
plt.show()
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.nn import PyroModule

# Reproducibility
pyro.clear_param_store()
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Synthetic time-series DGP: trend + seasonality + volatility + stronger noise
n = 800
t = np.arange(n, dtype=np.float32)
trend = 0.015 * t + 0.00002 * (t ** 2)
seasonality = 1.5 * np.sin(2 * np.pi * t / 40.0) + 0.8 * np.cos(2 * np.pi * t / 90.0)
clean_series = trend + seasonality

# Stronger time-varying volatility
volatility = (
    0.50
    + 0.50 * (np.sin(2 * np.pi * t / 120.0) ** 2)
    + 0.25 * (t / n)
)

noise = np.random.normal(loc=0.0, scale=volatility).astype(np.float32)
noisy_series = clean_series + noise
clean_series = clean_series.astype(np.float32)
noisy_series = noisy_series.astype(np.float32)

# Standardize using noisy series statistics
mean_y = noisy_series.mean()
std_y = noisy_series.std() + 1e-8

clean_std = (clean_series - mean_y) / std_y
noisy_std = (noisy_series - mean_y) / std_y

# Windowing for feature learning
window_size = 64
stride = 1

def make_windows(series: np.ndarray, window_size: int, stride: int = 1):
    xs = []
    starts = []
    for i in range(0, len(series) - window_size + 1, stride):
        xs.append(series[i:i + window_size])
        starts.append(i)
    return np.stack(xs).astype(np.float32), starts

clean_windows_np, starts = make_windows(clean_std, window_size=window_size, stride=stride)
noisy_windows_np, _ = make_windows(noisy_std, window_size=window_size, stride=stride)
clean_windows = torch.tensor(clean_windows_np, device=device)
noisy_windows = torch.tensor(noisy_windows_np, device=device)
input_dim = window_size
hidden_dim = 256
latent_dim = 16

# Deterministic autoencoder in Pyro: no priors, likelihood only
class TimeSeriesAutoencoder(PyroModule):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = PyroModule[nn.Sequential](
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = PyroModule[nn.Sequential](
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

model_net = TimeSeriesAutoencoder(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
).to(device)

def model(x_noisy: torch.Tensor, x_clean: torch.Tensor):
    pyro.module("autoencoder", model_net)
    x_hat = model_net(x_noisy)
    sigma = pyro.param(
        "sigma",
        lambda: torch.tensor(0.50, device=device),
        constraint=dist.constraints.positive,
    )
    with pyro.plate("windows", x_clean.shape[0]):
        pyro.sample("obs", dist.Normal(x_hat, sigma).to_event(1), obs=x_clean)

def guide(x_noisy: torch.Tensor, x_clean: torch.Tensor):
    return None

optimizer = Adam({"lr": 5e-4})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Mini-batch training
batch_size = 64
num_epochs = 700
n_windows = noisy_windows.shape[0]
loss_history = []

def iterate_minibatches(x_in: torch.Tensor, y_out: torch.Tensor, batch_size: int):
    idx = torch.randperm(x_in.shape[0], device=x_in.device)
    for start in range(0, x_in.shape[0], batch_size):
        batch_idx = idx[start:start + batch_size]
        yield x_in[batch_idx], y_out[batch_idx]


for epoch in range(num_epochs):
    running_loss = 0.0
    for xb, yb in iterate_minibatches(noisy_windows, clean_windows, batch_size):
        running_loss += svi.step(xb, yb)
    avg_loss = running_loss / n_windows
    loss_history.append(avg_loss)
    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1:4d} | Loss per window: {avg_loss:.6f} "
            f"| sigma={pyro.param('sigma').item():.4f}"
        )

# Decode / denoise all windows
with torch.no_grad():
    decoded_windows = model_net(noisy_windows).detach().cpu().numpy()
    latent_features = model_net.encode(noisy_windows).detach().cpu().numpy()
print("Latent feature matrix shape:", latent_features.shape)

# Reconstruct full time series from overlapping decoded windows
decoded_series_std = np.zeros(n, dtype=np.float32)
counts = np.zeros(n, dtype=np.float32)

for k, start in enumerate(starts):
    decoded_series_std[start:start + window_size] += decoded_windows[k]
    counts[start:start + window_size] += 1.0

decoded_series_std = decoded_series_std / np.maximum(counts, 1e-8)
decoded_series = decoded_series_std * std_y + mean_y

# Metrics
mse_noisy = np.mean((noisy_series - clean_series) ** 2)
mse_decoded = np.mean((decoded_series - clean_series) ** 2)
rmse_noisy = math.sqrt(mse_noisy)
rmse_decoded = math.sqrt(mse_decoded)
print(f"\nRMSE noisy   : {rmse_noisy:.6f}")
print(f"RMSE decoded : {rmse_decoded:.6f}")

# Plots
plt.figure(figsize=(12, 4))
plt.plot(t, clean_series, label="Original DGP (no noise)")
plt.plot(t, noisy_series, label="Observed noisy series", alpha=0.7)
plt.title("Synthetic Time Series: Trend + Seasonality + High Volatility")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(t, clean_series, label="Original DGP (no noise)")
plt.plot(t, decoded_series, label="Decoded time series")
plt.title("Original Time Series vs Decoded Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(t, clean_series, label="Original DGP (no noise)")
plt.plot(t, noisy_series, label="Observed noisy series", alpha=0.45)
plt.plot(t, decoded_series, label="Decoded time series")
plt.title("Original, Noisy, and Decoded Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("ELBO loss / window")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(t, noisy_series - clean_series, label="Observed noise")
plt.plot(t, decoded_series - clean_series, label="Decoded residual")
plt.title("Noise vs Residual After Autoencoder Decoding")
plt.xlabel("Time")
plt.ylabel("Deviation from clean signal")
plt.legend()
plt.tight_layout()
plt.show()
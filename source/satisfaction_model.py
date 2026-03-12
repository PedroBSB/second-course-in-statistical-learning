import pandas as pd
import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt

from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam

torch.set_default_dtype(torch.double)
pyro.set_rng_seed(0)
pyro.clear_param_store()

# 1) Load ACSI data
data_path = "data/acsi_data.csv"

df = pd.read_csv(data_path)

indicator_cols = ["SATIS", "CONFIRM", "IDEAL"]
data = df[indicator_cols].dropna().copy()

# Standardize for a numerically stable factor model
X_np = data.to_numpy(dtype=float)
X_np = (X_np - X_np.mean(axis=0, keepdims=True)) / X_np.std(axis=0, ddof=0, keepdims=True)

X = torch.tensor(X_np, dtype=torch.double)
n, p = X.shape

print(f"n = {n}, p = {p}")
print("Columns:", indicator_cols)

# 2) Bayesian one-factor model
def model(X):
    n, p = X.shape

    # Weakly informative Gaussian priors for intercepts
    alpha = pyro.sample(
        "alpha",
        dist.Normal(torch.zeros(p), 10.0 * torch.ones(p)).to_event(1)
    )

    # First loading fixed for identification; remaining loadings estimated
    lambda_free = pyro.sample(
        "lambda_free",
        dist.Normal(torch.zeros(p - 1), 10.0 * torch.ones(p - 1)).to_event(1)
    )
    # Handle both training (1D) and posterior sampling (2D) cases
    ones_shape = lambda_free.shape[:-1] + (1,)
    lam = torch.cat([torch.ones(ones_shape, dtype=X.dtype), lambda_free], dim=-1)

    # Gaussian prior on log standard deviations
    log_sigma = pyro.sample(
        "log_sigma",
        dist.Normal(torch.zeros(p), 2.0 * torch.ones(p)).to_event(1)
    )
    sigma = torch.exp(log_sigma)

    with pyro.plate("obs", n):
        eta = pyro.sample("eta", dist.Normal(0.0, 1.0))
        mean = alpha + eta.unsqueeze(-1) * lam
        pyro.sample("x", dist.Normal(mean, sigma).to_event(1), obs=X)


guide = AutoNormal(model)

# 3) Fit with SVI
optim = Adam({"lr": 0.02})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

num_steps = 4000
loss_history = []

for step in range(num_steps):
    loss = svi.step(X)
    loss_history.append(loss / n)

    if step % 500 == 0:
        print(f"step {step:4d} | loss per obs = {loss / n:.6f}")

# 4) Convergence plot
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("SVI iteration")
plt.ylabel("ELBO loss per observation")
plt.title("SVI convergence for Bayesian one-factor satisfaction model")
plt.grid(True, alpha=0.3)
plt.show()

# 5) Posterior samples and credible intervals
predictive = Predictive(
    model,
    guide=guide,
    num_samples=3000,
    return_sites=("alpha", "lambda_free", "log_sigma")
)

posterior = predictive(X)

# When Predictive is called with the full model, parameters may have extra dimensions
# We only want the parameter samples, not per-observation values
alpha_samps = posterior["alpha"]
lambda_free_samps = posterior["lambda_free"]
log_sigma_samps = posterior["log_sigma"]

# Handle potential extra dimensions from the plate
if alpha_samps.ndim == 3:
    # Shape is [S, n, p], we want just [S, p] by taking first observation
    alpha_samps = alpha_samps[:, 0, :]
if lambda_free_samps.ndim == 3:
    lambda_free_samps = lambda_free_samps[:, 0, :]
if log_sigma_samps.ndim == 3:
    log_sigma_samps = log_sigma_samps[:, 0, :]

lambda_samps = torch.cat(
    [torch.ones(lambda_free_samps.shape[0], 1, dtype=X.dtype), lambda_free_samps],
    dim=1
)
sigma_samps = torch.exp(log_sigma_samps)

def summarize(samples, names):
    q = torch.quantile(samples, torch.tensor([0.025, 0.5, 0.975], dtype=X.dtype), dim=0)
    out = pd.DataFrame({
        "parameter": names,
        "mean": samples.mean(dim=0).detach().cpu().numpy(),
        "sd": samples.std(dim=0).detach().cpu().numpy(),
        "q2.5": q[0].detach().cpu().numpy(),
        "q50": q[1].detach().cpu().numpy(),
        "q97.5": q[2].detach().cpu().numpy(),
    })
    return out

alpha_tbl = summarize(alpha_samps, [f"alpha_{c}" for c in indicator_cols])
lambda_tbl = summarize(lambda_samps, [f"lambda_{c}" for c in indicator_cols])
sigma_tbl = summarize(sigma_samps, [f"sigma_{c}" for c in indicator_cols])

print("\nIntercepts:")
print(alpha_tbl.round(4).to_string(index=False))

print("\nLoadings:")
print(lambda_tbl.round(4).to_string(index=False))

print("\nResidual standard deviations:")
print(sigma_tbl.round(4).to_string(index=False))

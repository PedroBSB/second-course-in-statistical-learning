import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist

from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from sklearn.model_selection import train_test_split

torch.set_default_dtype(torch.double)
pyro.set_rng_seed(123)
pyro.clear_param_store()

# 1) Load data
df = pd.read_csv("data/bfi.csv")

# Choose either a 1-factor or 2-factor setup
NUM_FACTORS = 2  # set to 1 or 2

if NUM_FACTORS == 1:
    item_cols = ["E1", "E2", "E3", "E4", "E5"]
elif NUM_FACTORS == 2:
    item_cols = ["E1", "E2", "E3", "E4", "E5", "N1", "N2", "N3", "N4", "N5"]
else:
    raise ValueError("NUM_FACTORS must be 1 or 2.")

data = df[item_cols].dropna().copy()

# Standardize items for numerical stability
X_np = data.to_numpy(dtype=float)
X_np = (X_np - X_np.mean(axis=0, keepdims=True)) / X_np.std(axis=0, ddof=0, keepdims=True)

X = torch.tensor(X_np, dtype=torch.double)
n, p = X.shape
K = NUM_FACTORS

# Train / validation split
X_train_np, X_val_np = train_test_split(X_np, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train_np, dtype=torch.double)
X_val = torch.tensor(X_val_np, dtype=torch.double)

# ------------------------------------------------------------
# Confirmatory loading structure mask
# For K=1: all E-items load on factor 1
# For K=2: E-items -> factor 1, N-items -> factor 2
# We also anchor one loading per factor at 1 for identification
# ------------------------------------------------------------
mask = torch.zeros((p, K), dtype=torch.double)

if K == 1:
    mask[:, 0] = 1.0
    anchor_rows = [(0, 0)]  # E1 loading fixed to 1
else:
    # E1-E5 load on factor 1
    mask[0:5, 0] = 1.0
    # N1-N5 load on factor 2
    mask[5:10, 1] = 1.0
    anchor_rows = [(0, 0), (5, 1)]  # E1=1 and N1=1

anchor_set = set(anchor_rows)
free_positions = []
for j in range(p):
    for k in range(K):
        if mask[j, k] == 1.0 and (j, k) not in anchor_set:
            free_positions.append((j, k))

n_free = len(free_positions)

# 2) Marginal factor model with Laplace prior on loadings
def build_lambda(lambda_free):
    Lambda = torch.zeros((p, K), dtype=torch.double)

    # fixed anchors
    for (j, k) in anchor_rows:
        Lambda[j, k] = 1.0

    # free loadings
    for idx, (j, k) in enumerate(free_positions):
        Lambda[j, k] = lambda_free[..., idx]

    return Lambda


def model(X, laplace_scale=0.3):
    n, p = X.shape

    alpha = pyro.sample(
        "alpha",
        dist.Normal(torch.zeros(p), 5.0 * torch.ones(p)).to_event(1)
    )

    lambda_free = pyro.sample(
        "lambda_free",
        dist.Laplace(
            torch.zeros(n_free),
            laplace_scale * torch.ones(n_free)
        ).to_event(1)
    )

    log_sigma = pyro.sample(
        "log_sigma",
        dist.Normal(torch.zeros(p), 1.0 * torch.ones(p)).to_event(1)
    )

    Lambda = build_lambda(lambda_free)
    sigma = torch.exp(log_sigma)

    Phi = torch.eye(K, dtype=torch.double)
    Theta_delta = torch.diag(sigma ** 2)
    Sigma = Lambda @ Phi @ Lambda.T + Theta_delta

    # small jitter for numerical stability
    Sigma = Sigma + 1e-5 * torch.eye(p, dtype=torch.double)

    with pyro.plate("obs", n):
        pyro.sample(
            "x",
            dist.MultivariateNormal(loc=alpha, covariance_matrix=Sigma),
            obs=X
        )


def fit_model(X_train, laplace_scale, lr=0.02, num_steps=3000):
    pyro.clear_param_store()

    guide = AutoNormal(model)
    optim = Adam({"lr": lr})
    svi = SVI(
        lambda X: model(X, laplace_scale=laplace_scale),
        guide,
        optim,
        loss=Trace_ELBO()
    )

    losses = []
    for step in range(num_steps):
        loss = svi.step(X_train)
        losses.append(loss / X_train.shape[0])

    return guide, losses


def posterior_summary(guide, X_ref, laplace_scale, num_samples=2000):
    predictive = Predictive(
        lambda X: model(X, laplace_scale=laplace_scale),
        guide=guide,
        num_samples=num_samples,
        return_sites=("alpha", "lambda_free", "log_sigma")
    )
    post = predictive(X_ref)

    alpha_s = post["alpha"]
    lambda_free_s = post["lambda_free"]
    log_sigma_s = post["log_sigma"]

    # reconstruct Lambda samples
    Lambda_s = []
    for s in range(lambda_free_s.shape[0]):
        Lambda_s.append(build_lambda(lambda_free_s[s]))
    Lambda_s = torch.stack(Lambda_s, dim=0)

    sigma_s = torch.exp(log_sigma_s)

    return {
        "alpha": alpha_s,
        "Lambda": Lambda_s,
        "sigma": sigma_s,
    }


def summarize_tensor(samples, row_names):
    q = torch.quantile(
        samples,
        torch.tensor([0.025, 0.5, 0.975], dtype=samples.dtype),
        dim=0
    )
    return pd.DataFrame({
        "parameter": row_names,
        "mean": samples.mean(0).detach().cpu().numpy().flatten(),
        "sd": samples.std(0).detach().cpu().numpy().flatten(),
        "q2.5": q[0].detach().cpu().numpy().flatten(),
        "q50": q[1].detach().cpu().numpy().flatten(),
        "q97.5": q[2].detach().cpu().numpy().flatten(),
    })


# 3) Hyperparameter tuning via validation split
candidate_scales = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]

results = []

for b in candidate_scales:
    guide, losses = fit_model(X_train, laplace_scale=b, lr=0.02, num_steps=2500)

    # Validation ELBO loss
    svi_eval = SVI(
        lambda X: model(X, laplace_scale=b),
        guide,
        Adam({"lr": 0.02}),
        loss=Trace_ELBO()
    )
    val_loss = svi_eval.evaluate_loss(X_val) / X_val.shape[0]

    results.append({
        "laplace_scale": b,
        "train_loss_last": losses[-1],
        "val_loss": val_loss,
        "guide": guide
    })
    print(f"Laplace scale={b:>4}: train_loss={losses[-1]:.4f}, val_loss={val_loss:.4f}")

best = min(results, key=lambda d: d["val_loss"])
best_b = best["laplace_scale"]
best_guide = best["guide"]

print("\nBest Laplace scale:", best_b)

# 4) Posterior summaries using the best hyperparameter
post = posterior_summary(best_guide, X_train, laplace_scale=best_b, num_samples=3000)

# Summarize Lambda
lambda_names = []
for j in range(p):
    for k in range(K):
        lambda_names.append(f"lambda[{item_cols[j]},{k+1}]")

Lambda_flat = post["Lambda"].reshape(post["Lambda"].shape[0], -1)
lambda_table = summarize_tensor(Lambda_flat, lambda_names)

# Summarize residual sigmas
sigma_names = [f"sigma[{c}]" for c in item_cols]
sigma_table = summarize_tensor(post["sigma"], sigma_names)

print("\nPosterior summary for loadings:")
print(lambda_table.round(4).to_string(index=False))

print("\nPosterior summary for residual standard deviations:")
print(sigma_table.round(4).to_string(index=False))
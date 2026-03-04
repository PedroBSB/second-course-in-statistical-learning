import math
import torch
import pandas as pd
import pyro.distributions as dist

# -----------------------------
# 1) Settings
# -----------------------------
seed = 123
device = "cpu"

T = 3000
batch_size = 64        # 1 (pure SGD), 64 (mini-batch), or n (full batch)
lr0 = 0.2
print_every = max(1, T // 20)

# For inference from SGD noise
burn_in = int(0.5 * T)       # start collecting gradients after burn-in
collect_every = 5            # thin gradient samples to reduce autocorrelation
eps_ridge = 1e-6             # numerical stability for matrix inversion
alpha = 0.05                 # 95% CI

# Reproducibility
torch.manual_seed(seed)

# -----------------------------
# 2) Read data from CSV
# -----------------------------
df = pd.read_csv("data/quadratic_model.csv")
n = len(df)

x = torch.tensor(df["x"].to_numpy(), dtype=torch.float32, device=device)  # (n,)
y = torch.tensor(df["y"].to_numpy(), dtype=torch.float32, device=device)  # (n,)
sigma_loaded = float(df["sigma_true"].iloc[0])  # constant in file

# -----------------------------
# 3) SGD loop + gradient collection for sandwich covariance
# -----------------------------
# Parameters theta = [a, b, c]
theta = torch.zeros(3, device=device, requires_grad=True)

loss_log = []
theta_log = []

# We'll collect mini-batch gradients near convergence to estimate J.
# J_hat ~ Cov( stochastic gradients ) at theta_hat
grad_samples = []

for t in range(T):
    # ---- Sample i_t or mini-batch B_t ----
    if batch_size >= n:
        idx = torch.arange(n, device=device)
    else:
        idx = torch.randint(low=0, high=n, size=(batch_size,), device=device)

    xb = x[idx]
    yb = y[idx]

    # ---- Compute stochastic gradient of the loss ----
    a, b, c = theta[0], theta[1], theta[2]
    mu_b = a * xb**2 + b * xb + c

    # Negative log-likelihood (mean over batch)
    loss_t = -dist.Normal(mu_b, sigma_loaded).log_prob(yb).mean()

    # Backprop to get gradient w.r.t theta
    if theta.grad is not None:
        theta.grad.zero_()
    loss_t.backward()

    # ---- Collect gradients after burn-in (for J estimation) ----
    if (t >= burn_in) and ((t - burn_in) % collect_every == 0):
        grad_samples.append(theta.grad.detach().cpu().clone())  # shape (3,)

    # ---- Update parameters ----
    eta_t = lr0 / math.sqrt(t + 1.0)
    with torch.no_grad():
        theta -= eta_t * theta.grad

    # ---- Logging ----
    if (t % print_every) == 0 or t == T - 1:
        loss_log.append(float(loss_t.detach().cpu().item()))
        theta_log.append(theta.detach().cpu().clone())
        print(
            f"t={t:4d}  eta={eta_t:.4f}  loss={loss_log[-1]:.4f}  "
            f"a={theta_log[-1][0]:.4f}  b={theta_log[-1][1]:.4f}  c={theta_log[-1][2]:.4f}"
        )

theta_hat = theta.detach().cpu().clone()
print("\nEstimated parameters (SGD):")
print({"a": float(theta_hat[0]), "b": float(theta_hat[1]), "c": float(theta_hat[2])})

if a_true is not None:
    print("\nTrue parameters (from CSV):")
    print({"a": a_true, "b": b_true, "c": c_true})

# -----------------------------
# 4) Estimate covariance via sandwich form using mini-batch gradients
#    V_hat ≈ (1/n) H_hat^{-1} J_hat H_hat^{-1}
# -----------------------------
# 4a) J_hat: covariance of stochastic gradients near convergence
G = torch.stack(grad_samples, dim=0)  # (m, 3)
m = G.shape[0]
if m < 10:
    raise RuntimeError(
        f"Not enough gradient samples to estimate covariance (got m={m}). "
        f"Increase T or reduce burn_in/collect_every."
    )

# Sample covariance of gradients: J_hat
G_centered = G - G.mean(dim=0, keepdim=True)
J_hat = (G_centered.T @ G_centered) / (m - 1)  # (3,3)

# 4b) H_hat: empirical Hessian of the *full* empirical risk at theta_hat
# We'll compute Hessian of the full-data negative log-likelihood mean.
theta_eval = theta.detach().clone().requires_grad_(True)

a, b, c = theta_eval[0], theta_eval[1], theta_eval[2]
mu_full = a * x**2 + b * x + c
loss_full = -dist.Normal(mu_full, sigma_loaded).log_prob(y).mean()

# Gradient and Hessian via autograd
grad_full = torch.autograd.grad(loss_full, theta_eval, create_graph=True)[0]  # (3,)
H_rows = []
for k in range(3):
    row_k = torch.autograd.grad(grad_full[k], theta_eval, retain_graph=True)[0]  # (3,)
    H_rows.append(row_k)
H_hat = torch.stack(H_rows, dim=0).detach().cpu()  # (3,3)

# 4c) Sandwich covariance estimate
I = torch.eye(3)
H_reg = H_hat + eps_ridge * I  # ridge for stability
H_inv = torch.linalg.inv(H_reg)

V_hat = (1.0 / n) * (H_inv @ J_hat @ H_inv)  # (3,3)
se = torch.sqrt(torch.diag(V_hat))

# -----------------------------
# 5) Confidence intervals (normal approximation)
# -----------------------------
# z_{1-alpha/2} for 95% CI
z = 1.959963984540054  # approx for 0.975 quantile

ci_lower = theta_hat - z * se
ci_upper = theta_hat + z * se

print("\nEstimated sandwich covariance V_hat (approx):")
print(V_hat.numpy())

print("\nStandard errors (approx):")
print({"se_a": float(se[0]), "se_b": float(se[1]), "se_c": float(se[2])})

print("\n95% Confidence intervals (approx):")
print(
    {
        "a": (float(ci_lower[0]), float(ci_upper[0])),
        "b": (float(ci_lower[1]), float(ci_upper[1])),
        "c": (float(ci_lower[2]), float(ci_upper[2])),
    }
)


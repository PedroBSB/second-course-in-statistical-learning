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
batch_size = 64
print_every = max(1, T // 20)

# ADAM hyperparameters
eta = 0.05          # global learning rate (ADAM)
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

# For inference from ADAM (serially correlated gradients)
burn_in = int(0.5 * T)
collect_every = 5
L = 20              # Newey--West truncation lag (tune as needed)
eps_ridge = 1e-6    # stabilizer for Hessian inversion
alpha = 0.05        # 95% CI

torch.manual_seed(seed)

# -----------------------------
# 2) Read data from CSV
# -----------------------------
df = pd.read_csv("data/quadratic_model.csv")
n = len(df)

x = torch.tensor(df["x"].to_numpy(), dtype=torch.float32, device=device)  # (n,)
y = torch.tensor(df["y"].to_numpy(), dtype=torch.float32, device=device)  # (n,)
sigma_loaded = float(df["sigma_true"].iloc[0])

# Optional: true params if stored
a_true = float(df["a_true"].iloc[0]) if "a_true" in df.columns else None
b_true = float(df["b_true"].iloc[0]) if "b_true" in df.columns else None
c_true = float(df["c_true"].iloc[0]) if "c_true" in df.columns else None

# -----------------------------
# 3) ADAM loop + score collection for HAC (Newey--West)
# -----------------------------
theta = torch.zeros(3, device=device, requires_grad=True)  # [a,b,c]

m_t = torch.zeros(3, device=device)
v_t = torch.zeros(3, device=device)

loss_log = []
theta_log = []

# Collect stochastic gradients s_t near convergence (serially correlated under ADAM)
score_samples = []

for t in range(1, T + 1):
    # ---- Sample mini-batch B_t ----
    if batch_size >= n:
        idx = torch.arange(n, device=device)
    else:
        idx = torch.randint(low=0, high=n, size=(batch_size,), device=device)

    xb = x[idx]
    yb = y[idx]

    # ---- Compute stochastic gradient h_t ----
    a, b, c = theta[0], theta[1], theta[2]
    mu_b = a * xb**2 + b * xb + c
    loss_t = -dist.Normal(mu_b, sigma_loaded).log_prob(yb).mean()

    if theta.grad is not None:
        theta.grad.zero_()
    loss_t.backward()
    h_t = theta.grad.detach().clone()  # gradient vector (3,)

    # ---- Collect score/gradient samples after burn-in ----
    if (t >= burn_in) and ((t - burn_in) % collect_every == 0):
        score_samples.append(h_t.detach().cpu().clone())

    # ---- ADAM moment updates ----
    m_t = beta1 * m_t + (1.0 - beta1) * h_t
    v_t = beta2 * v_t + (1.0 - beta2) * (h_t * h_t)

    # bias correction
    m_hat = m_t / (1.0 - beta1 ** t)
    v_hat = v_t / (1.0 - beta2 ** t)

    # ---- Parameter update ----
    with torch.no_grad():
        theta -= eta * m_hat / (torch.sqrt(v_hat) + eps)

    # ---- Logging ----
    if (t % print_every) == 0 or t == T:
        loss_log.append(float(loss_t.detach().cpu().item()))
        theta_log.append(theta.detach().cpu().clone())
        print(
            f"t={t:4d}  loss={loss_log[-1]:.4f}  "
            f"a={theta_log[-1][0]:.4f}  b={theta_log[-1][1]:.4f}  c={theta_log[-1][2]:.4f}"
        )

theta_hat = theta.detach().cpu().clone()
print("\nEstimated parameters (ADAM):")
print({"a": float(theta_hat[0]), "b": float(theta_hat[1]), "c": float(theta_hat[2])})

if a_true is not None:
    print("\nTrue parameters (from CSV):")
    print({"a": a_true, "b": b_true, "c": c_true})

# -----------------------------
# 4) Newey--West estimator for Sigma (HAC) from score_samples
#    Sigma_NW = Gamma_0 + sum_{l=1}^L w_l (Gamma_l + Gamma_l')
#    Gamma_l = (1/Ts) sum_{t=l+1}^{Ts} s_t s_{t-l}'
# -----------------------------
S = torch.stack(score_samples, dim=0)  # (Ts, 3)
Ts = S.shape[0]
if Ts < (L + 5):
    raise RuntimeError(
        f"Not enough score samples for Newey--West with L={L} (Ts={Ts}). "
        f"Increase T, reduce L, reduce burn_in, or reduce collect_every."
    )

# Center the scores (recommended for HAC)
S_centered = S - S.mean(dim=0, keepdim=True)

def bartlett_weight(l, L):
    return 1.0 - (l / (L + 1.0))

# Gamma_0
Gamma0 = (S_centered.T @ S_centered) / Ts  # (3,3)

Sigma_NW = Gamma0.clone()

for l in range(1, L + 1):
    w_l = bartlett_weight(l, L)
    # Gamma_l = (1/Ts) sum_{t=l}^{Ts-1} s_t s_{t-l}^T  (0-indexed)
    A = S_centered[l:, :]          # s_t
    B = S_centered[:-l, :]         # s_{t-l}
    Gamma_l = (A.T @ B) / Ts
    Sigma_NW = Sigma_NW + w_l * (Gamma_l + Gamma_l.T)

Sigma_NW = Sigma_NW.detach().cpu()

# -----------------------------
# 5) H_hat (full-data Hessian at theta_hat) and sandwich covariance
#    V_hat ≈ (1/n) H_hat^{-1} Sigma_NW H_hat^{-1}
# -----------------------------
theta_eval = theta.detach().clone().requires_grad_(True)

a, b, c = theta_eval[0], theta_eval[1], theta_eval[2]
mu_full = a * x**2 + b * x + c
loss_full = -dist.Normal(mu_full, sigma_loaded).log_prob(y).mean()

grad_full = torch.autograd.grad(loss_full, theta_eval, create_graph=True)[0]
H_rows = []
for k in range(3):
    row_k = torch.autograd.grad(grad_full[k], theta_eval, retain_graph=True)[0]
    H_rows.append(row_k)
H_hat = torch.stack(H_rows, dim=0).detach().cpu()

I = torch.eye(3)
H_reg = H_hat + eps_ridge * I
H_inv = torch.linalg.inv(H_reg)

V_hat = (1.0 / n) * (H_inv @ Sigma_NW @ H_inv)
se = torch.sqrt(torch.diag(V_hat))

# -----------------------------
# 6) Confidence intervals (normal approximation)
# -----------------------------
z = 1.959963984540054  # 97.5% quantile of N(0,1)
ci_lower = theta_hat - z * se
ci_upper = theta_hat + z * se

print("\nEstimated HAC (Newey--West) Sigma_NW:")
print(Sigma_NW.numpy())

print("\nEstimated sandwich covariance V_hat (HAC):")
print(V_hat.numpy())

print("\nStandard errors (HAC):")
print({"se_a": float(se[0]), "se_b": float(se[1]), "se_c": float(se[2])})

print("\n95% Confidence intervals (HAC):")
print(
    {
        "a": (float(ci_lower[0]), float(ci_upper[0])),
        "b": (float(ci_lower[1]), float(ci_upper[1])),
        "c": (float(ci_lower[2]), float(ci_upper[2])),
    }
)

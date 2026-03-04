import math
import torch
import pandas as pd
import pyro.distributions as dist

# 1) Settings
seed = 123
device = "cpu"

T = 3000
batch_size = 64        # try 1 (pure SGD), 64 (mini-batch), or n (full batch)
lr0 = 0.2              # base learning rate
print_every = max(1, T // 20)


# 2) Read data from CSV
df = pd.read_csv("data/quadratic_model.csv")
n = len(df)
x = torch.tensor(df["x"].to_numpy(), dtype=torch.float32, device=device)  # (n,)
y = torch.tensor(df["y"].to_numpy(), dtype=torch.float32, device=device)  # (n,)

# Use sigma from file (constant, so read first row)
sigma_loaded = float(df["sigma_true"].iloc[0])

# 3) Define model + loss (NLL under Normal) and run explicit SGD loop

# Parameters theta = [a, b, c]
theta = torch.zeros(3, device=device, requires_grad=True)

# Learning-rate schedule eta_t = lr0 / sqrt(t+1)
loss_log = []
theta_log = []

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

theta_hat = {"a": float(theta[0].item()), "b": float(theta[1].item()), "c": float(theta[2].item())}

print("\nEstimated parameters (SGD):")
print(theta_hat)
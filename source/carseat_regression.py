import pandas as pd
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt

pyro.set_rng_seed(0)
pyro.clear_param_store()

# ---------- Load data ----------
df = pd.read_csv("data/Carseats.csv")

y_col = "Sales"
x_cols = ["CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education"]

cols = [y_col] + x_cols
data = df[cols].dropna()

Y = torch.tensor(data[y_col].to_numpy(), dtype=torch.double)
X = torch.tensor(data[x_cols].to_numpy(), dtype=torch.double)

n, p = X.shape  # p = 7

# ---------- Sample covariance S of observed vector z = [y, x1, ..., xp] ----------
Z = torch.cat([Y[:, None], X], dim=1)  # [n, 1+p]
Zc = Z - Z.mean(dim=0, keepdim=True)
S = (Zc.T @ Zc) / n  # MLE covariance

# ---------- Helpers ----------
def make_pd_from_A(A: torch.Tensor, jitter: float = 1e-4) -> torch.Tensor:
    return A @ A.T + jitter * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

def sem_implied_cov(beta: torch.Tensor, Phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """
    SEM:
      y = beta^T x + zeta
      x ~ (0, Phi),  zeta ~ (0, psi),  Cov(x, zeta)=0

    Joint covariance of z = [y; x]:
      Sigma_x  = Phi
      Sigma_yx = beta^T Phi
      Sigma_yy = beta^T Phi beta + psi
    """
    Sigma_x = Phi                                   # [p,p]
    Sigma_yx = beta.reshape(1, p) @ Phi             # [1,p]
    Sigma_xy = Sigma_yx.T                           # [p,1]
    Sigma_yy = (beta @ (Phi @ beta)) + psi          # scalar

    Sigma = torch.zeros((1 + p, 1 + p), dtype=Phi.dtype, device=Phi.device)
    Sigma[0, 0] = Sigma_yy
    Sigma[0, 1:] = Sigma_yx
    Sigma[1:, 0] = Sigma_xy.squeeze(-1)
    Sigma[1:, 1:] = Sigma_x
    return Sigma

def gaussian_cov_fit_loss(S: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
    """
    ℓ_Gaussian(θ) = 1/2 [ log det Σ(θ) + tr( S Σ(θ)^{-1} ) ]
    """
    sign, logdet = torch.slogdet(Sigma)
    if sign <= 0:
        return torch.tensor(float("inf"), dtype=Sigma.dtype, device=Sigma.device)

    # tr(S Sigma^{-1}) = tr( solve(Sigma, S) )
    Sigma_inv_S = torch.linalg.solve(Sigma, S)
    trace_term = torch.trace(Sigma_inv_S)
    return 0.5 * (logdet + trace_term)

# ---------- Pyro model ----------
def model(S: torch.Tensor, n: int):
    beta = pyro.param("beta", torch.zeros(p, dtype=torch.double))

    A = pyro.param("A", 0.1 * torch.eye(p, dtype=torch.double))
    Phi = make_pd_from_A(A, jitter=1e-4)

    psi_unconstrained = pyro.param("psi_unconstrained", torch.tensor(0.0, dtype=torch.double))
    psi = torch.nn.functional.softplus(psi_unconstrained) + 1e-6

    Sigma = sem_implied_cov(beta=beta, Phi=Phi, psi=psi)
    loss = gaussian_cov_fit_loss(S=S, Sigma=Sigma)

    # scale by n (Gaussian likelihood up to constants)
    pyro.factor("cov_fit", -n * loss)

def guide(S: torch.Tensor, n: int):
    return None

# ---------- Fit with SVI + record loss ----------
optim = Adam({"lr": 5e-2})
svi = SVI(model=model, guide=guide, optim=optim, loss=Trace_ELBO())

num_steps = 3000
loss_history = []

for step in range(num_steps):
    svi.step(S, n)

    # record the objective value ℓ_Gaussian(θ)
    with torch.no_grad():
        beta = pyro.param("beta")
        Phi = make_pd_from_A(pyro.param("A"))
        psi = torch.nn.functional.softplus(pyro.param("psi_unconstrained")) + 1e-6
        Sigma = sem_implied_cov(beta, Phi, psi)
        cur_loss = gaussian_cov_fit_loss(S, Sigma).item()
    loss_history.append(cur_loss)

    if step % 250 == 0:
        print(f"step {step:4d} | loss={cur_loss:.6f}")

# ---------- Plot convergence ----------
plt.figure()
plt.plot(loss_history)
plt.xlabel("SVI iteration")
plt.ylabel(r"$\ell_{\mathrm{Gaussian}}(\boldsymbol{\theta})$")
plt.title("Loss convergence (covariance-fit objective)")
plt.grid(True, alpha=0.3)
plt.show()
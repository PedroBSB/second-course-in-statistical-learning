import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.double)
torch.manual_seed(0)

# -----------------------------
# 1) Load data + normalize column names + keep numerical only
# -----------------------------
df = pd.read_csv("data/CreditCard.csv")

# Normalize column names to avoid KeyError due to spaces / casing
df.columns = (
    df.columns.astype(str)
      .str.strip()
      .str.lower()
      .str.replace(r"\s+", "_", regex=True)
)

# Keep numeric columns only
df = df.select_dtypes(include=[np.number]).dropna().copy()

# 2) Define SEM roles (in SEM notation: y, eta, xi, z)
# Outcome (endogenous variable)
Y_COL = "expenditure"

# Endogenous regressor (a component of endogenous system; needs instruments)
ENDO_COL = "share"

# Exogenous controls (xi)
# NOTE: we keep the list but automatically intersect with df.columns
X_COLS = ['income', 'age', 'reports', 'dependents', 'active']

# Excluded instruments (must be in data; should NOT be in X_COLS)
# Here: months and majorcards are typical excluded shifters
Z_EXCL = ["months", "majorcards"]

# 3) Build tensors
y = torch.tensor(df[Y_COL].to_numpy())                # [n]
share = torch.tensor(df[ENDO_COL].to_numpy())         # [n]
X = torch.tensor(df[X_COLS].to_numpy())               # [n, k]
Z_excl = torch.tensor(df[Z_EXCL].to_numpy())          # [n, l]

n = y.shape[0]
k = X.shape[1]
l = Z_excl.shape[1]

ones = torch.ones((n, 1))

# Structural regressors (SEM: eta depends on share + xi)
# x_struct_i = [1, share_i, X_i]
X_struct = torch.cat([ones, share.reshape(-1, 1), X], dim=1)  # [n, p]
p = X_struct.shape[1]

# Instrument matrix (SEM/GMM: z_i includes intercept + exogenous controls + excluded instruments)
# Standard IV: use all exogenous variables as instruments + excluded instruments
Z_inst = torch.cat([ones, X, Z_excl], dim=1)                   # [n, q]
q = Z_inst.shape[1]

# 4) Identification sanity checks
# (a) Instrument rank
with torch.no_grad():
    svals = torch.linalg.svdvals(Z_inst)
    tol = 1e-10 * svals.max()
    rankZ = int((svals > tol).sum().item())
print(f"Z_inst shape={tuple(Z_inst.shape)}, rank={rankZ} (max {q})")

# (b) Order condition: number of instruments >= number of endogenous regressors (here: 1) + intercept
# In practice for IV, we want at least as many excluded instruments as endogenous regressors.
print(f"Excluded instruments count = {l} (endogenous regressors needing instruments = 1)")

# 5) One-step GMM objective (SEM moment fitting)
def moment_vector(beta: torch.Tensor) -> torch.Tensor:
    """
    m_hat(beta) = (1/n) Σ z_i * u_i(beta)
    Returns vector in R^q
    """
    resid = y - (X_struct @ beta)                    # [n]
    m = (Z_inst * resid.reshape(-1, 1)).mean(dim=0)  # [q]
    return m

def gmm_loss(beta: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    J(beta) = m_hat(beta)' W m_hat(beta)
    """
    m = moment_vector(beta)
    return m @ W @ m

# Start with identity weighting (one-step GMM)
W = torch.eye(q)

beta = torch.nn.Parameter(torch.zeros(p))
opt = torch.optim.Adam([beta], lr=1e-2)

loss_hist = []
num_steps = 5000

for t in range(num_steps):
    opt.zero_grad()
    loss = gmm_loss(beta, W)
    loss.backward()
    opt.step()

    loss_hist.append(loss.item())
    if t % 500 == 0:
        print(f"step {t:4d} | J(beta)={loss.item():.6e}")


beta_hat = beta.detach().cpu().numpy()

# 6) Report estimates (SEM interpretation)
names_struct = ["Intercept", ENDO_COL] + X_COLS
print("\nEstimated structural coefficients (GMM / SEM moments):")
for name, val in zip(names_struct, beta_hat):
    print(f"{name:>12s}: {val: .6f}")

# 8) Plot convergence
plt.figure()
plt.plot(loss_hist, label="1-step GMM (W=I)")
plt.xlabel("Iteration")
plt.ylabel(r"$J(\beta)=\hat m(\beta)^\top W \hat m(\beta)$")
plt.title("GMM loss convergence (SEM moment fitting)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
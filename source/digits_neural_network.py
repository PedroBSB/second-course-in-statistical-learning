import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
pyro.set_rng_seed(42)

# 1.  DATA  (REQUIRE: {(x_i, y_i)}_{i=1}^n)
digits = datasets.load_digits()
X_raw, y_raw = digits.data.astype(np.float32), digits.target.astype(np.int64)

# Standard scaling — BN also normalizes internally, but input scaling helps
# the first layer receive well-conditioned activations.
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_tr_np, X_te_np, y_tr_np, y_te_np = train_test_split(
    X_scaled, y_raw, test_size=0.2, stratify=y_raw, random_state=42
)

X_train = torch.tensor(X_tr_np)
y_train = torch.tensor(y_tr_np)
X_test  = torch.tensor(X_te_np)
y_test  = torch.tensor(y_te_np)

n, d_in   = X_train.shape   # 1437, 64
d_out     = 10               # classes 0–9

# 2.  HYPERPARAMETERS  (REQUIRE: η, θ^(0))
eta        = 1e-3   # learning rate η
batch_size = 64     # |B|
max_epochs = 300    # upper bound on the Repeat loop

hidden_widths = [128, 64]
layer_sizes   = [d_in] + hidden_widths + [d_out]   # [64, 128, 64, 10]
L             = len(hidden_widths)

# 3.  NETWORK DEFINITION  —  two hidden layers, each followed by BatchNorm1d
class BNNetwork(nn.Module):
    """
    Fully-connected MLP with BatchNorm1d after every hidden layer.
    REQUIRE: θ^(0) — PyTorch default Kaiming init on Linear layers.
    """

    def __init__(self, layer_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        for l_idx, (in_f, out_f) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:]), start=1
        ):
            # Linear transform: z^(l) = W^(l) h + b^(l)
            layers.append(nn.Linear(in_f, out_f))

            if l_idx < len(layer_sizes) - 1:     # hidden layers only
                # BatchNorm1d: implicit regularization (replaces λ_n P(θ))
                # eps=1e-5, momentum=0.1 (fraction of new batch stat to blend)
                layers.append(nn.BatchNorm1d(out_f, eps=1e-5, momentum=0.1))
                layers.append(nn.ReLU())
            # No BN / activation after the output layer

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# REQUIRE: θ^(0) — network initialized at t = 0
network = BNNetwork(layer_sizes)

# 4.  PYRO MODEL
def model(X_batch: torch.Tensor, y_batch: torch.Tensor) -> None:
    # Register all nn.Module parameters (W, b, γ_BN, β_BN) into Pyro param store.
    # pyro.module also handles train/eval mode toggling for running stats.
    net = pyro.module("network", network)

    # ── FORWARD PASS
    #   For each i in B:
    #     h_i^(1) = ReLU( BN( W^(1) x_i + b^(1) ) )   ← BN acts here
    #     h_i^(2) = ReLU( BN( W^(2) h_i^(1) + b^(2) ) )
    #     ŷ_i     = W^(3) h_i^(2) + b^(3)              ← output (logits)
    logits = net(X_batch)   # shape (|B|, 10)

    # ── LOSS EVALUATION
    #   R_B(θ) = (1/|B|) Σ_{i∈B} ℓ(y_i, ŷ_i)
    #   encoded as the plate log-likelihood (cross-entropy)
    with pyro.plate("data", X_batch.shape[0]):
        pyro.sample("y", dist.Categorical(logits=logits), obs=y_batch)

# 5.  GUIDE  —  deterministic delta posterior  q(θ) = δ(θ − θ̂)
def guide(X_batch: torch.Tensor, y_batch: torch.Tensor) -> None:
    """Empty guide: all parameters live in the model via pyro.module."""
    pass

# 6.  SVI SETUP
pyro.clear_param_store()
optimizer = ClippedAdam({"lr": eta, "clip_norm": 5.0})
svi       = SVI(model, guide, optimizer, loss=Trace_ELBO())

# 7.  INFERENCE HELPERS
def predict(X: torch.Tensor) -> torch.Tensor:
    """Prediction in eval mode: BN uses running stats (not batch stats)."""
    network.eval()
    with torch.no_grad():
        return network(X).argmax(dim=1)

def accuracy(X: torch.Tensor, y: torch.Tensor) -> float:
    return (predict(X) == y).float().mean().item()

# 8.  TRAINING LOOP
t         = 0               # line 1: t ← 0
prev_loss = float("inf")
tol       = 1e-4
loss_history: list[float] = []

print("=" * 62)
print(f"  BN Backprop via Pyro SVI  |  η={eta}  |B|={batch_size}")
print(f"  Architecture : {layer_sizes}")
print(f"  Regularizer  : BatchNorm1d (implicit, no λ_n term)")
print("=" * 62)
print(f"{'Epoch':>6}  {'−ELBO':>12}  {'Train acc':>10}  {'Test acc':>9}")
print("-" * 46)

while t < max_epochs:     # REPEAT

    # ── Line 3: Sample mini-batch B ⊂ {1, …, n} ──────────────────────────────
    idx     = torch.randperm(n)[:batch_size]
    X_batch = X_train[idx]
    y_batch = y_train[idx]

    # ── Lines 4–20: forward + loss + backward + update via SVI.step ──────────
    # network.train() is set internally by pyro.module during SVI.step
    neg_elbo = svi.step(X_batch, y_batch)

    loss_history.append(neg_elbo)
    t += 1   # line 21: t ← t + 1

    if t % 30 == 0:
        tr_acc = accuracy(X_train, y_train)
        te_acc = accuracy(X_test,  y_test)
        print(f"{t:>6}  {neg_elbo:>12.2f}  {tr_acc:>9.3f}  {te_acc:>8.3f}")
        network.train()   # restore train mode after accuracy eval

    # ── Convergence criterion: relative change in −ELBO < tol ─────────────────
    if t > 10:
        rel_change = abs(prev_loss - neg_elbo) / (abs(prev_loss) + 1e-8)
        if rel_change < tol:
            print(f"\n  Converged at epoch {t}  (Δloss = {rel_change:.2e} < {tol})")
            break

    prev_loss = neg_elbo

# 9.  RESULTS
theta_hat = {
    name: param.detach().clone()
    for name, param in pyro.get_param_store().items()
}

final_train = accuracy(X_train, y_train)
final_test  = accuracy(X_test,  y_test)

print("\n" + "=" * 62)
print("  θ̂ retrieved — MAP estimates with implicit BN regularization")
for name, tensor in theta_hat.items():
    print(f"    {name:<40}  shape={tuple(tensor.shape)}")
print(f"\n  Final train accuracy : {final_train:.3%}")
print(f"  Final test  accuracy : {final_test:.3%}")
print(f"  Stopped at epoch     : {t}")
print("=" * 62)
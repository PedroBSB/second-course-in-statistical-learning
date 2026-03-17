import torch
from pyro.nn import PyroModule, PyroParam
class HybridPSpline(PyroModule):
    def __init__(self, n_linear_features: int, n_spline_features: int):
        super().__init__()
        self.theta = PyroParam(torch.zeros(n_linear_features, dtype=torch.float32))
        self.beta = PyroParam(torch.zeros(n_spline_features, dtype=torch.float32))

    def forward(self, x_linear: torch.Tensor, x_spline: torch.Tensor) -> torch.Tensor:
        return x_linear @ self.theta + x_spline @ self.beta

import pyro
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from source.model_hybrid_spline import HybridPSpline
from source.preprocessing_hybrid_spline import build_difference_matrix, to_tensor

def fit_hybrid_pspline(
    x_linear_train: np.ndarray,
    x_spline_train: np.ndarray,
    y_train: np.ndarray,
    smoothing_lambda: float,
    lr: float = 0.03,
    num_steps: int = 3000,
) -> tuple[HybridPSpline, list[float]]:
    pyro.clear_param_store()
    model = HybridPSpline(
        n_linear_features=x_linear_train.shape[1],
        n_spline_features=x_spline_train.shape[1],
    )

    x_linear_t = to_tensor(x_linear_train)
    x_spline_t = to_tensor(x_spline_train)
    y_t = to_tensor(y_train).reshape(-1)
    diff_matrix = build_difference_matrix(x_spline_train.shape[1], order=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        y_hat = model(x_linear_t, x_spline_t)
        mse_loss = torch.mean((y_t - y_hat) ** 2)
        roughness = torch.mean((diff_matrix @ model.beta) ** 2)
        loss = mse_loss + smoothing_lambda * roughness
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    return model, history

def predict_hybrid_pspline(model: HybridPSpline, x_linear: np.ndarray, x_spline: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        preds = model(to_tensor(x_linear), to_tensor(x_spline)).detach().cpu().numpy()
    return preds

def validation_mse(
    model: HybridPSpline,
    x_linear_val: np.ndarray,
    x_spline_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    preds = predict_hybrid_pspline(model, x_linear_val, x_spline_val)
    return float(mean_squared_error(y_val, preds))

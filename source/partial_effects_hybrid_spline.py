import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.compose import ColumnTransformer

from source.config_hybrid_spline import (
    DATA_DIR,
    DEVICE,
    LINEAR_COLS,
    SPLINE_COL,
)
from source.model_hybrid_spline import HybridPSpline
from source.preprocessing_hybrid_spline import split_design_matrix

PARTIAL_EFFECT_PLOT_PATH = DATA_DIR / "hybrid_pyro_pspline_partial_effects.png"


def get_linear_partial_effects(model: HybridPSpline, preprocessor: ColumnTransformer) -> pd.DataFrame:
    """
    Extract partial effects for the linear (categorical) covariates.

    For a linear component y = X_linear @ theta, the partial effect of covariate j
    is simply theta_j. Because the categorical variable is one-hot encoded with 'first'
    dropped, each coefficient represents the differential effect relative to the
    baseline category.
    """
    feature_names = preprocessor.get_feature_names_out()
    linear_names = [n for n in feature_names if n.startswith("model_")]
    theta = model.theta.detach().cpu().numpy()

    rows = [{"covariate": "model_baseline (reference)", "partial_effect": 0.0}]
    for name, coef in zip(linear_names, theta):
        rows.append({"covariate": name, "partial_effect": float(coef)})

    return pd.DataFrame(rows)


def get_spline_partial_effects(
    model: HybridPSpline,
    preprocessor: ColumnTransformer,
    times_range: tuple[float, float],
    n_grid: int = 400,
    eps: float = 1e-4,
    reference_category: str = "baseline",
) -> pd.DataFrame:
    """
    Compute the partial effect of `times` on the response via numerical
    differentiation of the fitted spline surface.

    The partial effect at a point t is approximated by the central difference:

        df/dt ≈ [f(t + eps) - f(t - eps)] / (2 * eps)

    We evaluate using a fixed linear design (reference category) so the
    derivative isolates the spline component only.
    """
    grid = np.linspace(times_range[0], times_range[1], n_grid)

    grid_plus = grid + eps
    grid_minus = grid - eps

    def _predict_at(times_values: np.ndarray) -> np.ndarray:
        df_grid = pd.DataFrame({
            "times": times_values,
            "model": reference_category,
        })
        design = preprocessor.transform(df_grid[[*LINEAR_COLS, SPLINE_COL]])
        feature_names = preprocessor.get_feature_names_out()
        x_linear, x_spline = split_design_matrix(design, feature_names)
        with torch.no_grad():
            x_lin_t = torch.tensor(x_linear, dtype=torch.float32, device=DEVICE)
            x_spl_t = torch.tensor(x_spline, dtype=torch.float32, device=DEVICE)
            preds = model(x_lin_t, x_spl_t).detach().cpu().numpy()
        return preds

    pred_plus = _predict_at(grid_plus)
    pred_minus = _predict_at(grid_minus)
    partial_effect = (pred_plus - pred_minus) / (2 * eps)

    fitted = _predict_at(grid)

    return pd.DataFrame({
        "times": grid,
        "fitted_value": fitted,
        "partial_effect": partial_effect,
    })


def plot_partial_effects(
    linear_effects: pd.DataFrame,
    spline_effects: pd.DataFrame,
) -> None:
    """
    Two-panel figure:
      Left  – partial effect of `times` (numerical derivative of spline curve)
      Right – partial effects of categorical `model` levels (bar chart of theta)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left panel: spline partial effect of times ---
    ax = axes[0]
    ax.plot(
        spline_effects["times"],
        spline_effects["partial_effect"],
        linewidth=2.0,
        color="steelblue",
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("times (ms after impact)")
    ax.set_ylabel(r"$\partial\, \hat{f} \,/\, \partial\, \mathrm{times}$")
    ax.set_title("Partial effect of times (spline component)")
    ax.grid(alpha=0.25)

    # --- Right panel: linear partial effects (bar chart) ---
    ax = axes[1]
    covariates = linear_effects["covariate"].tolist()
    effects = linear_effects["partial_effect"].tolist()
    colors = ["#aaaaaa" if e == 0.0 else "steelblue" for e in effects]
    bars = ax.barh(covariates, effects, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, effects):
        offset = 0.3 if val >= 0 else -0.3
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}",
            va="center",
            fontsize=10,
        )
    ax.set_xlabel(r"$\theta_j$ (partial effect on accel)")
    ax.set_title("Partial effects of model (linear component)")
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle("Partial Effects — Hybrid P-Spline Model", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(PARTIAL_EFFECT_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_partial_effects(
    linear_effects: pd.DataFrame,
    spline_effects: pd.DataFrame,
) -> None:
    """Print raw numbers for both partial-effect tables."""
    print("\n" + "=" * 60)
    print("PARTIAL EFFECTS — LINEAR COMPONENT (model)")
    print("=" * 60)
    print(
        linear_effects.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        )
    )

    print("\n" + "=" * 60)
    print("PARTIAL EFFECTS — SPLINE COMPONENT (times)")
    print("=" * 60)

    # Show a summary at key quantile positions
    quantile_indices = np.linspace(0, len(spline_effects) - 1, 20, dtype=int)
    summary = spline_effects.iloc[quantile_indices].copy()
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(f"\nPartial effect of times — range: "
          f"[{spline_effects['partial_effect'].min():.4f}, "
          f"{spline_effects['partial_effect'].max():.4f}]")
    print(f"Mean absolute partial effect: "
          f"{spline_effects['partial_effect'].abs().mean():.4f}")
    print("=" * 60)

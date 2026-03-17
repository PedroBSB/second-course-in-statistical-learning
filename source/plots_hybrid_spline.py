import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from source.config_hybrid_spline import *
from source.preprocessing_hybrid_spline import split_design_matrix
from source.training_hybrid_spline import predict_hybrid_pspline
from source.model_hybrid_spline import HybridPSpline


def plot_final_fit(df_all: pd.DataFrame, preprocessor: ColumnTransformer, model: HybridPSpline) -> None:
    categories = sorted(df_all['model'].unique().tolist())
    grid_times = np.linspace(df_all['times'].min(), df_all['times'].max(), 400)

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {'baseline': 'o', 'sport': 's', 'touring': '^'}

    for category in categories:
        subset = df_all[df_all['model'] == category]
        ax.scatter(
            subset['times'],
            subset['accel'],
            alpha=0.70,
            label=f'{category} observed',
            marker=markers.get(category, 'o'),
        )

        grid_df = pd.DataFrame({'times': grid_times, 'model': category})
        design = preprocessor.transform(grid_df[[*LINEAR_COLS, SPLINE_COL]])
        feature_names = preprocessor.get_feature_names_out()
        x_linear_grid, x_spline_grid = split_design_matrix(design, feature_names)
        grid_pred = predict_hybrid_pspline(model, x_linear_grid, x_spline_grid)
        ax.plot(grid_times, grid_pred, linewidth=2.0, label=f'{category} fitted')

    ax.set_title('Hybrid Pyro P-spline fit on mcycle data with mock model effect')
    ax.set_xlabel('times (ms after impact)')
    ax.set_ylabel('accel (g)')
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIT_PLOT_PATH, dpi=200)
    plt.close(fig)

def plot_lambda_risk(candidate_df: pd.DataFrame) -> None:
    grouped = (
        candidate_df.groupby(['n_knots', 'lambda'], as_index=False)
        .agg(
            mean_inner_mse=('inner_mse_mean', 'mean'),
            std_inner_mse=('inner_mse_mean', 'std'),
        )
        .sort_values(['n_knots', 'lambda'])
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for n_knots, subset in grouped.groupby('n_knots'):
        x = subset['lambda'].to_numpy(dtype=float)
        y = subset['mean_inner_mse'].to_numpy(dtype=float)
        s = subset['std_inner_mse'].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, marker='o', label=f'n_knots={n_knots}')
        ax.fill_between(x, y - s, y + s, alpha=0.15)

    ax.set_xscale('log')
    ax.set_xlabel('lambda')
    ax.set_ylabel('Mean validation MSE')
    ax.set_title('Validation risk versus smoothing parameter')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LAMBDA_PLOT_PATH, dpi=200)
    plt.close(fig)

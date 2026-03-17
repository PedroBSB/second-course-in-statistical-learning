from pathlib import Path
import numpy as np
import torch
import pyro

SEED = 42
pyro.set_rng_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "mcycle_hybrid.csv"
FIT_PLOT_PATH = DATA_DIR / "hybrid_pyro_pspline_fit.png"
LAMBDA_PLOT_PATH = DATA_DIR / "lambda_risk_plot.png"
METRICS_PATH = DATA_DIR / "hybrid_pyro_pspline_metrics.csv"
PREDICTIONS_PATH = DATA_DIR / "hybrid_pyro_pspline_predictions.csv"
CANDIDATE_SCORES_PATH = DATA_DIR / "hybrid_pyro_pspline_candidate_scores.csv"

TARGET_COL = 'accel'
LINEAR_COLS = ['model']
SPLINE_COL = 'times'
TEST_SIZE = 0.20
OUTER_FOLDS = 5
INNER_FOLDS = 3
DEVICE = torch.device('cpu')

PARAM_GRID = [
    {'n_knots': 8, 'lambda': 1e-4, 'lr': 0.03, 'steps': 2500},
    {'n_knots': 8, 'lambda': 1e-3, 'lr': 0.03, 'steps': 2500},
    {'n_knots': 10, 'lambda': 1e-3, 'lr': 0.03, 'steps': 3000},
    {'n_knots': 10, 'lambda': 1e-2, 'lr': 0.03, 'steps': 3000},
    {'n_knots': 12, 'lambda': 1e-2, 'lr': 0.02, 'steps': 3500},
    {'n_knots': 12, 'lambda': 1e-1, 'lr': 0.02, 'steps': 3500},
]

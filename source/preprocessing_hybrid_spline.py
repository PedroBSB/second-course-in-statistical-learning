import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from source.config_hybrid_spline import *

def build_difference_matrix(n_spline_features: int, order: int = 2) -> torch.Tensor:
    if order < 1:
        raise ValueError('order must be >= 1')
    eye = np.eye(n_spline_features, dtype=np.float32)
    diff = np.diff(eye, n=order, axis=0)
    return torch.tensor(diff, dtype=torch.float32, device=DEVICE)

def build_preprocessor(n_knots: int) -> ColumnTransformer:
    spline = SplineTransformer(
        n_knots=n_knots,
        degree=3,
        knots='quantile',
        include_bias=False,
        extrapolation='linear',
    )
    return ColumnTransformer(
        transformers=[
            ('linear', OneHotEncoder(drop='first', sparse_output=False), LINEAR_COLS),
            ('spline', spline, [SPLINE_COL]),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )

def split_design_matrix(design: np.ndarray, feature_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    linear_mask = np.array([name.startswith('model_') for name in feature_names])
    spline_mask = np.array([
        name.startswith('times_sp_') or name.startswith('times_spline') or name.startswith('times_')
        for name in feature_names
    ])
    if not linear_mask.any():
        raise ValueError(f'No linear model features were generated. Feature names: {feature_names.tolist()}')
    if not spline_mask.any():
        raise ValueError(f'No spline features were generated. Feature names: {feature_names.tolist()}')
    return design[:, linear_mask], design[:, spline_mask]


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=DEVICE)

def prepare_train_test_designs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_knots: int,
) -> tuple[ColumnTransformer, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = build_preprocessor(n_knots=n_knots)
    train_design = preprocessor.fit_transform(train_df[[*LINEAR_COLS, SPLINE_COL]])
    test_design = preprocessor.transform(test_df[[*LINEAR_COLS, SPLINE_COL]])
    feature_names = preprocessor.get_feature_names_out()
    train_linear, train_spline = split_design_matrix(train_design, feature_names)
    test_linear, test_spline = split_design_matrix(test_design, feature_names)
    return preprocessor, train_linear, train_spline, test_linear, test_spline

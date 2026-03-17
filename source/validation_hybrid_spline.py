import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from source.config_hybrid_spline  import INNER_FOLDS, LINEAR_COLS, OUTER_FOLDS, SEED, SPLINE_COL, TARGET_COL
from source.preprocessing_hybrid_spline import build_preprocessor, split_design_matrix
from source.training_hybrid_spline import fit_hybrid_pspline, predict_hybrid_pspline


def evaluate_configuration(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_knots: int,
    smoothing_lambda: float,
    lr: float,
    num_steps: int,
) -> float:
    preprocessor = build_preprocessor(n_knots=n_knots)
    x_train_design = preprocessor.fit_transform(train_df[[*LINEAR_COLS, SPLINE_COL]])
    x_val_design = preprocessor.transform(val_df[[*LINEAR_COLS, SPLINE_COL]])
    feature_names = preprocessor.get_feature_names_out()

    x_linear_train, x_spline_train = split_design_matrix(x_train_design, feature_names)
    x_linear_val, x_spline_val = split_design_matrix(x_val_design, feature_names)

    model, _ = fit_hybrid_pspline(
        x_linear_train=x_linear_train,
        x_spline_train=x_spline_train,
        y_train=train_df[TARGET_COL].to_numpy(),
        smoothing_lambda=smoothing_lambda,
        lr=lr,
        num_steps=num_steps,
    )
    preds = predict_hybrid_pspline(model, x_linear_val, x_spline_val)
    return float(mean_squared_error(val_df[TARGET_COL].to_numpy(), preds))



def nested_cross_validation(
    train_df: pd.DataFrame,
    param_grid: list[dict],
    outer_folds: int = OUTER_FOLDS,
    inner_folds: int = INNER_FOLDS,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=SEED)
    outer_rows: list[dict] = []
    all_candidate_scores: list[dict] = []

    for outer_fold, (outer_tr_idx, outer_val_idx) in enumerate(outer_cv.split(train_df), start=1):
        outer_train = train_df.iloc[outer_tr_idx].reset_index(drop=True)
        outer_val = train_df.iloc[outer_val_idx].reset_index(drop=True)

        inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=SEED + outer_fold)
        candidate_scores: list[dict] = []

        for params in param_grid:
            inner_mses: list[float] = []
            for inner_tr_idx, inner_val_idx in inner_cv.split(outer_train):
                inner_train = outer_train.iloc[inner_tr_idx].reset_index(drop=True)
                inner_val = outer_train.iloc[inner_val_idx].reset_index(drop=True)
                mse = evaluate_configuration(
                    train_df=inner_train,
                    val_df=inner_val,
                    n_knots=params['n_knots'],
                    smoothing_lambda=params['lambda'],
                    lr=params['lr'],
                    num_steps=params['steps'],
                )
                inner_mses.append(mse)

            row = {
                **params,
                'outer_fold': outer_fold,
                'inner_mse_mean': float(np.mean(inner_mses)),
                'inner_mse_std': float(np.std(inner_mses)),
            }
            candidate_scores.append(row)
            all_candidate_scores.append(row)

        best_params = min(candidate_scores, key=lambda row: row['inner_mse_mean'])

        preprocessor = build_preprocessor(n_knots=best_params['n_knots'])
        x_outer_train_design = preprocessor.fit_transform(outer_train[[*LINEAR_COLS, SPLINE_COL]])
        x_outer_val_design = preprocessor.transform(outer_val[[*LINEAR_COLS, SPLINE_COL]])
        feature_names = preprocessor.get_feature_names_out()
        x_linear_train, x_spline_train = split_design_matrix(x_outer_train_design, feature_names)
        x_linear_val, x_spline_val = split_design_matrix(x_outer_val_design, feature_names)

        model, _ = fit_hybrid_pspline(
            x_linear_train=x_linear_train,
            x_spline_train=x_spline_train,
            y_train=outer_train[TARGET_COL].to_numpy(),
            smoothing_lambda=best_params['lambda'],
            lr=best_params['lr'],
            num_steps=best_params['steps'],
        )
        outer_preds = predict_hybrid_pspline(model, x_linear_val, x_spline_val)
        outer_rows.append({
            'outer_fold': outer_fold,
            'best_n_knots': best_params['n_knots'],
            'best_lambda': best_params['lambda'],
            'best_lr': best_params['lr'],
            'best_steps': best_params['steps'],
            'outer_mse': float(mean_squared_error(outer_val[TARGET_COL].to_numpy(), outer_preds)),
            'outer_rmse': float(math.sqrt(mean_squared_error(outer_val[TARGET_COL].to_numpy(), outer_preds))),
            'outer_r2': float(r2_score(outer_val[TARGET_COL].to_numpy(), outer_preds)),
        })

    summary = pd.DataFrame(outer_rows)
    candidate_df = pd.DataFrame(all_candidate_scores)
    best_overall = (
        summary.groupby(['best_n_knots', 'best_lambda', 'best_lr', 'best_steps'], as_index=False)
        .agg(mean_outer_mse=('outer_mse', 'mean'))
        .sort_values('mean_outer_mse', ascending=True)
        .iloc[0]
        .to_dict()
    )
    final_params = {
        'n_knots': int(best_overall['best_n_knots']),
        'lambda': float(best_overall['best_lambda']),
        'lr': float(best_overall['best_lr']),
        'steps': int(best_overall['best_steps']),
    }
    return summary, final_params, candidate_df

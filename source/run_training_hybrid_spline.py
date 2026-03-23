import math
import warnings
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings(
    'ignore',
    message=r'.*invalid escape sequence.*',
    category=SyntaxWarning,
)

from source.config_hybrid_spline import *
from source.partial_effects_hybrid_spline import (
    get_linear_partial_effects,
    get_spline_partial_effects,
    plot_partial_effects,
    print_partial_effects,
)
from source.plots_hybrid_spline import plot_final_fit, plot_lambda_risk
from source.preprocessing_hybrid_spline import prepare_train_test_designs
from source.training_hybrid_spline import fit_hybrid_pspline, predict_hybrid_pspline
from source.validation_hybrid_spline import nested_cross_validation


df = pd.read_csv(DATA_PATH)
df = df.dropna().copy()
df['model'] = df['model'].astype('category')

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SEED,
    shuffle=True,
    stratify=df['model'],
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

nested_cv_results, best_params, candidate_scores_df = nested_cross_validation(
    train_df=train_df,
    param_grid=PARAM_GRID,
)

candidate_scores_df.to_csv(CANDIDATE_SCORES_PATH, index=False)
plot_lambda_risk(candidate_scores_df)

final_preprocessor, train_linear, train_spline, test_linear, test_spline = prepare_train_test_designs(
    train_df=train_df,
    test_df=test_df,
    n_knots=best_params['n_knots'],
)

final_model, training_history = fit_hybrid_pspline(
    x_linear_train=train_linear,
    x_spline_train=train_spline,
    y_train=train_df[TARGET_COL].to_numpy(),
    smoothing_lambda=best_params['lambda'],
    lr=best_params['lr'],
    num_steps=best_params['steps'],
)

test_pred = predict_hybrid_pspline(final_model, test_linear, test_spline)
train_pred = predict_hybrid_pspline(final_model, train_linear, train_spline)

metrics_df = nested_cv_results.copy()
metrics_df['final_best_n_knots'] = best_params['n_knots']
metrics_df['final_best_lambda'] = best_params['lambda']
metrics_df['final_best_lr'] = best_params['lr']
metrics_df['final_best_steps'] = best_params['steps']
metrics_df['holdout_train_rmse'] = float(math.sqrt(mean_squared_error(train_df[TARGET_COL], train_pred)))
metrics_df['holdout_test_rmse'] = float(math.sqrt(mean_squared_error(test_df[TARGET_COL], test_pred)))
metrics_df['holdout_test_r2'] = float(r2_score(test_df[TARGET_COL], test_pred))
metrics_df.to_csv(METRICS_PATH, index=False)

predictions_df = test_df.copy()
predictions_df['pred_accel'] = test_pred
predictions_df['residual'] = predictions_df['accel'] - predictions_df['pred_accel']
predictions_df.to_csv(PREDICTIONS_PATH, index=False)

plot_final_fit(df_all=df, preprocessor=final_preprocessor, model=final_model)

# ── Partial effects ──────────────────────────────────────────
linear_effects = get_linear_partial_effects(model=final_model, preprocessor=final_preprocessor)
spline_effects = get_spline_partial_effects(
    model=final_model,
    preprocessor=final_preprocessor,
    times_range=(df['times'].min(), df['times'].max()),
)

print_partial_effects(linear_effects, spline_effects)
plot_partial_effects(linear_effects, spline_effects)

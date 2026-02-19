import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def generate_data(n: int = 100, noise: float = 10.0, seed: int = 42) -> tuple:
    """Generate synthetic linear regression data."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 100, size=(n, 1))
    true_slope, true_intercept = 2.5, 15.0
    y = true_slope * X.squeeze() + true_intercept + rng.normal(0, noise, size=n)
    return X, y


def fit_and_evaluate(X: np.ndarray, y: np.ndarray) -> dict:
    """Fit a linear regression model and return evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "coef": model.coef_[0],
        "intercept": model.intercept_,
    }


def plot_results(results: dict, X: np.ndarray, y: np.ndarray) -> None:
    """Plot the data and fitted regression line."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot with regression line
    ax = axes[0]
    ax.scatter(X, y, alpha=0.5, label="Data", color="steelblue")
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    ax.plot(x_line, results["model"].predict(x_line), color="crimson", lw=2,
            label=f"Fit: y = {results['coef']:.2f}x + {results['intercept']:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Linear Regression Fit")
    ax.legend()

    # Residuals plot
    ax = axes[1]
    residuals = results["y_test"] - results["y_pred"]
    ax.scatter(results["y_pred"], residuals, alpha=0.6, color="darkorange")
    ax.axhline(0, color="black", lw=1, linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals  (RMSE={results['rmse']:.2f}, R²={results['r2']:.3f})")

    plt.tight_layout()
    plt.savefig("images/linear_regression_results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    X, y = generate_data(n=200, noise=12.0)
    results = fit_and_evaluate(X, y)

    print(f"Coefficient : {results['coef']:.4f}")
    print(f"Intercept   : {results['intercept']:.4f}")
    print(f"RMSE        : {results['rmse']:.4f}")
    print(f"R²          : {results['r2']:.4f}")

    plot_results(results, X, y)

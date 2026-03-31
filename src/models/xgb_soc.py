from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def train_xgboost_soc(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    *,
    random_state: int = 42,
    n_splits: int = 3,
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """
    XGBoost regressor with TimeSeriesSplit CV over training data.
    """
    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
        "n_estimators": [100, 200],
    }
    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        # Avoid joblib+XGBoost thread explosion / worker crashes on laptops.
        n_jobs=1,
        tree_method="hist",
    )
    tscv = TimeSeriesSplit(n_splits=n_splits)
    grid = GridSearchCV(
        base,
        param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        # Prevent loky worker crashes (TerminatedWorkerError) from parallel CV.
        n_jobs=1,
        verbose=0,
    )
    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - t0

    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # inference latency (ms/sample) on test set
    t_inf0 = time.perf_counter()
    _ = model.predict(X_test.iloc[: min(500, len(X_test))])
    t_inf1 = time.perf_counter()
    n_inf = min(500, len(X_test))
    latency_ms = (t_inf1 - t_inf0) / max(n_inf, 1) * 1000.0

    metrics = {
        "best_params": grid.best_params_,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
        "training_time_sec": train_time_sec,
        "inference_latency_ms_per_sample": latency_ms,
        "cv_best_score_neg_mse": float(grid.best_score_),
    }
    return model, metrics


def save_shap_summary(
    model: xgb.XGBRegressor,
    X_sample: pd.DataFrame,
    out_path: Path,
    *,
    max_samples: int = 2000,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Xs = X_sample
    if len(Xs) > max_samples:
        Xs = Xs.sample(n=max_samples, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)
    shap.summary_plot(shap_values, Xs, show=False)
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def persist_xgb(model: xgb.XGBRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_xgb(path: Path) -> xgb.XGBRegressor:
    return joblib.load(path)


def metrics_to_json(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _jsonable(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items()}
        if isinstance(x, (np.floating, np.integer, float, int)):
            return float(x)
        return x

    serializable = {k: _jsonable(v) for k, v in metrics.items()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

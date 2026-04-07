from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly.bms_anomaly import build_anomaly_flags_csv
from src.features.soc_features import build_soc_dataset, time_based_split_per_device
from src.ingestion.pipeline import parse_ev_payload
from src.models.lstm_soc import save_lstm_bundle, train_lstm_soc
from src.models.xgb_soc import metrics_to_json as xgb_metrics_to_json
from src.models.xgb_soc import persist_xgb, save_shap_summary, train_xgboost_soc


def _save_comparison_table_png(comparison: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfp = comparison.copy()

    for col in ["XGBoost", "LSTM"]:
        if col in dfp.columns:
            dfp[col] = dfp[col].apply(lambda x: f"{x:.4g}" if isinstance(x, (float, int, np.floating, np.integer)) else x)

    fig_w = 12
    fig_h = max(3.5, 0.6 + 0.45 * len(dfp))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=dfp.values,
        colLabels=dfp.columns.tolist(),
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.35)

    header_color = "#1B5E20"
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#1B5E20")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 else "#E8F5E9")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    models_dir = Path("models")
    out_dir = Path("outputs/task2")
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_ev_payload("data/raw/ev_prod_data.csv")
    ds = build_soc_dataset(df, horizon_minutes=10, sequence_cadence_seconds=30)
    is_test = time_based_split_per_device(ds.meta, test_ratio=0.2)

    export = pd.concat(
        [
            ds.meta,
            ds.X,
            ds.y.rename("target_soc_t_plus_10min"),
            is_test.rename("is_test").astype(int),
        ],
        axis=1,
    )
    export.to_csv(out_dir / "soc_features.csv", index=False)

    train_mask = ~is_test.values
    test_mask = is_test.values

    train_meta = ds.meta.loc[train_mask].reset_index(drop=True)
    X_train = ds.X.loc[train_mask].reset_index(drop=True)
    y_train = ds.y.loc[train_mask].reset_index(drop=True)

    test_meta = ds.meta.loc[test_mask].reset_index(drop=True)
    X_test = ds.X.loc[test_mask].reset_index(drop=True)
    y_test = ds.y.loc[test_mask].reset_index(drop=True)

    sort_tr = train_meta["ts"].argsort()
    X_train = X_train.iloc[sort_tr].reset_index(drop=True)
    y_train = y_train.iloc[sort_tr].reset_index(drop=True)
    train_meta = train_meta.iloc[sort_tr].reset_index(drop=True)

    sort_te = test_meta["ts"].argsort()
    X_test = X_test.iloc[sort_te].reset_index(drop=True)
    y_test = y_test.iloc[sort_te].reset_index(drop=True)
    test_meta = test_meta.iloc[sort_te].reset_index(drop=True)

    # --- XGBoost ---
    xgb_model, xgb_metrics = train_xgboost_soc(
        X_train,
        y_train,
        X_test,
        y_test,
        list(ds.feature_columns),
        meta_train=train_meta,
    )
    persist_xgb(xgb_model, models_dir / "xgboost_soc.pkl")
    xgb_metrics_to_json(xgb_metrics, out_dir / "xgboost_metrics.json")
    save_shap_summary(xgb_model, X_test, out_dir / "shap_xgboost_summary.png")

    # --- LSTM ---
    order_tr = train_meta.sort_values(["device_id", "ts"], kind="mergesort").index
    order_te = test_meta.sort_values(["device_id", "ts"], kind="mergesort").index
    X_train_lstm = X_train.loc[order_tr].reset_index(drop=True)
    y_train_lstm = y_train.loc[order_tr].reset_index(drop=True)
    train_meta_lstm = train_meta.loc[order_tr].reset_index(drop=True)
    X_test_lstm = X_test.loc[order_te].reset_index(drop=True)
    y_test_lstm = y_test.loc[order_te].reset_index(drop=True)
    test_meta_lstm = test_meta.loc[order_te].reset_index(drop=True)

    lstm_model, lstm_scaler, lstm_metrics = train_lstm_soc(
        X_train_lstm.values.astype(np.float32),
        y_train_lstm.values.astype(np.float32),
        train_meta_lstm["device_id"].values,
        X_test_lstm.values.astype(np.float32),
        y_test_lstm.values.astype(np.float32),
        test_meta_lstm["device_id"].values,
        plot_dir=out_dir,
        seq_len=10,
    )
    save_lstm_bundle(
        lstm_model,
        lstm_scaler,
        n_features=len(ds.feature_columns),
        path=models_dir / "lstm_soc.pt",
    )
    with (out_dir / "lstm_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(lstm_metrics, f, indent=2)

    # --- Comparison table ---
    comparison = pd.DataFrame(
        {
            "Metric": [
                "Test RMSE (SoC %)",
                "Test MAE (SoC %)",
                "Training time (sec)",
                "Inference latency (ms/sample)",
                "Interpretability (1–5)",
                "Production readiness (1–5)",
            ],
            "XGBoost": [
                round(xgb_metrics["test_rmse"], 4),
                round(xgb_metrics["test_mae"], 4),
                round(xgb_metrics["training_time_sec"], 2),
                round(xgb_metrics["inference_latency_ms_per_sample"], 4),
                4,
                5,
            ],
            "LSTM": [
                round(lstm_metrics["test_rmse"], 4),
                round(lstm_metrics["test_mae"], 4),
                round(lstm_metrics["training_time_sec"], 2),
                round(lstm_metrics["inference_latency_ms_per_sample"], 4),
                2,
                3,
            ],
            "Winner_and_justification": [
                "Lower is better; compare RMSE/MAE",
                "Lower is better",
                "Faster training often XGBoost on tabular",
                "Lower is better for online inference",
                "XGBoost: SHAP + tree structure",
                "XGBoost: simpler ops deploy; LSTM needs seq buffer",
            ],
        }
    )
    comparison.to_csv(out_dir / "model_comparison_table.csv", index=False)
    _save_comparison_table_png(comparison, out_dir / "model_comparison_table.png")

    # --- Anomalies ---
    flags = build_anomaly_flags_csv(df)
    flags.to_csv("anomaly_flags.csv", index=False)
    flags.to_csv(out_dir / "anomaly_flags.csv", index=False)

    print("Task 2 artifacts:")
    print(f"  {out_dir / 'soc_features.csv'}")
    print(f"  {models_dir / 'xgboost_soc.pkl'}")
    print(f"  {out_dir / 'shap_xgboost_summary.png'}")
    print(f"  {models_dir / 'lstm_soc.pt'}")
    print(f"  {out_dir / 'model_comparison_table.csv'}")
    print(f"  {out_dir / 'model_comparison_table.png'}")
    print("  anomaly_flags.csv")


if __name__ == "__main__":
    main()

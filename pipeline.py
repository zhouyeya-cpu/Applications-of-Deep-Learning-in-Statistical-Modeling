"""
Unified tabular regression benchmark pipeline - Energy Efficiency dataset.

Right-click run version:
- In VS Code, open this file and click "Run Python File".
- No terminal arguments are needed.

Dataset:
- UCI Energy Efficiency dataset, id=242
- 768 samples, 8 numeric building-design features
- Targets: Y1 = Heating Load, Y2 = Cooling Load
- This script uses Y1 by default. Change TARGET_COLUMN below if needed.

Models:
1. Linear Regression
2. Ridge Regression
3. SVR-RBF
4. Gaussian Process Regression
5. Random Forest
6. HistGradientBoosting
7. XGBoost if installed
8. MLP
9. FT-Transformer, simple PyTorch implementation

Outputs:
benchmark_outputs_energy_efficiency/
    results/
        model_results.csv
        test_predictions.csv
        test_indices.csv
    figures/
        01_r2_sorted.png
        02_mae_rmse_grouped.png
        03_training_time_log.png
        04_predicted_vs_true_grid.png
        05_sorted_test_predictions.png
        06_residual_boxplot.png
        07_error_vs_true_best_model.png

Install if needed:
pip install numpy pandas scikit-learn matplotlib ucimlrepo torch
Optional:
pip install xgboost
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# =========================
# User settings
# =========================
RANDOM_STATE = 42
TARGET_COLUMN = "Y1"  # "Y1" = Heating Load, "Y2" = Cooling Load
TARGET_NAME = "Heating Load" if TARGET_COLUMN == "Y1" else "Cooling Load"
OUTPUT_DIR = Path("benchmark_outputs_energy_efficiency")
TEST_SIZE = 0.20
CV_FOLDS = 5

# Set this to True when debugging. For paper results, keep False.
QUICK_MODE = False


FEATURE_RENAME = {
    "X1": "Relative Compactness",
    "X2": "Surface Area",
    "X3": "Wall Area",
    "X4": "Roof Area",
    "X5": "Overall Height",
    "X6": "Orientation",
    "X7": "Glazing Area",
    "X8": "Glazing Area Distribution",
    "Y1": "Heating Load",
    "Y2": "Cooling Load",
}


@dataclass
class BenchmarkResult:
    dataset: str
    domain: str
    target: str
    model: str
    n_train_used: int
    mae: float
    rmse: float
    r2: float
    train_time_sec: float
    best_params: str


def set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compatible with old and new sklearn versions."""
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_energy_efficiency_dataset() -> pd.DataFrame:
    """Load UCI Energy Efficiency dataset using ucimlrepo."""
    try:
        from ucimlrepo import fetch_ucirepo
    except Exception as exc:
        raise RuntimeError(
            "Missing package: ucimlrepo. Install it first:\n"
            "pip install ucimlrepo\n"
            "Then right-click run this file again."
        ) from exc

    dataset = fetch_ucirepo(id=242)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    df = pd.concat([X, y], axis=1)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def prepare_energy_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Prepare X and y for Energy Efficiency regression."""
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found. Available columns: {list(df.columns)}")

    # Drop both targets from features to avoid leakage.
    drop_cols = [c for c in ["Y1", "Y2"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col].astype(float).values

    # This dataset is numeric, but keep this general.
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    X = X[numeric_cols]
    return X, y, numeric_cols


def build_preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
    ], remainder="drop")


class FTTransformerRegressor(BaseEstimator, RegressorMixin):
    """Small FT-Transformer style regressor for preprocessed tabular data.

    Each feature becomes a token:
        token_j = x_j * weight_j + bias_j
    A CLS token is added. Transformer encoder predicts from the CLS output.
    """

    def __init__(
        self,
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        epochs: int = 200,
        patience: int = 25,
        random_state: int = RANDOM_STATE,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for FTTransformerRegressor. Install: pip install torch")
        set_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        idx_train, idx_val = train_test_split(
            np.arange(len(X)), test_size=0.15, random_state=self.random_state
        )
        X_train, y_train = X[idx_train], y[idx_train]
        X_val, y_val = X[idx_val], y[idx_val]

        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_x = torch.tensor(X_val).to(self.device_)
        val_y = torch.tensor(y_val).to(self.device_)

        self.model_ = _FTTransformerNet(
            n_features=self.n_features_in_,
            d_token=self.d_token,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device_)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        patience_left = self.patience

        for epoch in range(self.epochs):
            self.model_.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad()
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_pred = self.model_(val_x)
                val_loss = loss_fn(val_pred, val_y).item()

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"FT-Transformer epoch {epoch + 1}: val_loss={val_loss:.6f}")

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                xb = torch.tensor(X[start:start + self.batch_size]).to(self.device_)
                pred = self.model_(xb).detach().cpu().numpy().reshape(-1)
                preds.append(pred)
        return np.concatenate(preds)


class _FTTransformerNet(nn.Module):
    def __init__(self, n_features: int, d_token: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.encoder(tokens)
        return self.head(out[:, 0, :])


def get_models() -> Dict[str, object]:
    """Model registry. All models use the same fixed train/test split."""
    models: Dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": GridSearchCV(
            Ridge(),
            param_grid={"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
        ),
        "SVR-RBF": GridSearchCV(
            SVR(kernel="rbf"),
            param_grid={
                "C": [1.0, 10.0, 50.0, 100.0] if not QUICK_MODE else [1.0, 10.0],
                "gamma": ["scale", 0.01, 0.05, 0.1],
                "epsilon": [0.01, 0.1, 0.2],
            },
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
        ),
        "Gaussian Process Regression": GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
            alpha=1e-6,
            normalize_y=True,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300 if not QUICK_MODE else 100,
            max_depth=None,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=400 if not QUICK_MODE else 120,
            max_leaf_nodes=31,
            l2_regularization=0.001,
            random_state=RANDOM_STATE,
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=600 if not QUICK_MODE else 150,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400 if not QUICK_MODE else 120,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if TORCH_AVAILABLE:
        models["FT-Transformer"] = FTTransformerRegressor(
            d_token=32,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            lr=1e-3,
            batch_size=64,
            epochs=200 if not QUICK_MODE else 60,
            patience=25 if not QUICK_MODE else 10,
            random_state=RANDOM_STATE,
        )

    return models


def fit_predict_one_model(
    model_name: str,
    model,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    numeric_cols: List[str],
) -> Tuple[BenchmarkResult, np.ndarray]:
    preprocessor = build_preprocessor(numeric_cols)

    start = time.time()
    if model_name == "FT-Transformer":
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        best_params = {
            "d_token": model.d_token,
            "n_heads": model.n_heads,
            "n_layers": model.n_layers,
            "dropout": model.dropout,
            "lr": model.lr,
            "epochs": model.epochs,
        }
    else:
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])
        pipe.fit(X_train_raw, y_train)
        pred = pipe.predict(X_test_raw)
        if isinstance(model, GridSearchCV):
            best_params = model.best_params_
        else:
            best_params = {}
    train_time = time.time() - start

    mae = mean_absolute_error(y_test, pred)
    rmse = rmse_score(y_test, pred)
    r2 = r2_score(y_test, pred)

    result = BenchmarkResult(
        dataset="UCI Energy Efficiency",
        domain="Energy / Climate / Built Environment",
        target=TARGET_NAME,
        model=model_name,
        n_train_used=len(X_train_raw),
        mae=float(mae),
        rmse=float(rmse),
        r2=float(r2),
        train_time_sec=float(train_time),
        best_params=json.dumps(best_params),
    )
    return result, pred


def add_value_labels(ax, fmt="{:.3f}", rotation=0):
    """Add labels above bars."""
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        x = patch.get_x() + patch.get_width() / 2
        ax.annotate(
            fmt.format(height),
            (x, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=rotation,
        )


def plot_results(
    results_df: pd.DataFrame,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results_df = results_df.sort_values("r2", ascending=False).reset_index(drop=True)
    model_order = results_df["model"].tolist()

    # 1. R2 sorted bar with labels.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(results_df["model"], results_df["r2"])
    ax.set_title(f"R² Comparison on Fixed Test Set ({TARGET_NAME})")
    ax.set_ylabel("R² Score, higher is better")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    add_value_labels(ax, fmt="{:.3f}")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_r2_sorted.png", dpi=220)
    plt.close(fig)

    # 2. MAE and RMSE grouped bar, normalized by best score for readability.
    metric_df = results_df.copy()
    metric_df["MAE / best MAE"] = metric_df["mae"] / metric_df["mae"].min()
    metric_df["RMSE / best RMSE"] = metric_df["rmse"] / metric_df["rmse"].min()
    x = np.arange(len(metric_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, metric_df["MAE / best MAE"], width, label="MAE / best MAE")
    ax.bar(x + width / 2, metric_df["RMSE / best RMSE"], width, label="RMSE / best RMSE")
    ax.set_title("Normalized Error Comparison on Fixed Test Set")
    ax.set_ylabel("Relative error, lower is better")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_df["model"], rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "02_mae_rmse_grouped.png", dpi=220)
    plt.close(fig)

    # 3. Training time on log scale. This makes fast and slow models visible together.
    time_df = results_df.sort_values("train_time_sec", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(time_df["model"], time_df["train_time_sec"])
    ax.set_yscale("log")
    ax.set_title("Training Time Comparison, Log Scale")
    ax.set_ylabel("Training time seconds, log scale")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    for patch in ax.patches:
        height = patch.get_height()
        x_pos = patch.get_x() + patch.get_width() / 2
        ax.annotate(f"{height:.2f}s", (x_pos, height), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_training_time_log.png", dpi=220)
    plt.close(fig)

    # 4. Predicted vs true grid. This directly shows how each model behaves.
    n_models = len(model_order)
    n_cols = 3
    n_rows = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.6 * n_rows))
    axes = np.array(axes).reshape(-1)
    min_v = min([y_true.min()] + [predictions[m].min() for m in model_order])
    max_v = max([y_true.max()] + [predictions[m].max() for m in model_order])
    for i, model_name in enumerate(model_order):
        ax = axes[i]
        pred = predictions[model_name]
        row = results_df[results_df["model"] == model_name].iloc[0]
        ax.scatter(y_true, pred, s=18, alpha=0.65)
        ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1)
        ax.set_title(f"{model_name}\nR²={row['r2']:.3f}, RMSE={row['rmse']:.3f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
    for j in range(n_models, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Predicted vs True on the Same Test Set ({TARGET_NAME})", y=1.02, fontsize=16)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_predicted_vs_true_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 5. Sorted test predictions. The same fixed test samples are sorted by true y.
    sorted_idx = np.argsort(y_true)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(y_true[sorted_idx], linewidth=2.5, label="True", marker="o", markersize=3)
    # Show all if not too many; otherwise top models only.
    shown_models = model_order[:min(6, len(model_order))]
    for model_name in shown_models:
        ax.plot(predictions[model_name][sorted_idx], linewidth=1.5, alpha=0.85, label=model_name)
    ax.set_title("Same Test Samples Sorted by True Target")
    ax.set_xlabel("Test sample index after sorting by true value")
    ax.set_ylabel(TARGET_NAME)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "05_sorted_test_predictions.png", dpi=220)
    plt.close(fig)

    # 6. Residual boxplot. Lower spread around 0 is better.
    residuals = [predictions[m] - y_true for m in model_order]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.boxplot(residuals, labels=model_order, showfliers=True)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Residual Distribution on Fixed Test Set")
    ax.set_ylabel("Prediction error = predicted - true")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(fig_dir / "06_residual_boxplot.png", dpi=220)
    plt.close(fig)

    # 7. Error vs true for the best model.
    best_model = model_order[0]
    best_pred = predictions[best_model]
    abs_error = np.abs(best_pred - y_true)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(y_true, abs_error, s=22, alpha=0.65)
    ax.set_title(f"Absolute Error vs True Target: {best_model}")
    ax.set_xlabel(f"True {TARGET_NAME}")
    ax.set_ylabel("Absolute error")
    fig.tight_layout()
    fig.savefig(fig_dir / "07_error_vs_true_best_model.png", dpi=220)
    plt.close(fig)


def run_benchmark() -> None:
    set_seed(RANDOM_STATE)
    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Unified Tabular Regression Benchmark")
    print("Dataset: UCI Energy Efficiency, id=242")
    print(f"Target: {TARGET_COLUMN} = {TARGET_NAME}")
    print("Same fixed train/test split is used for every model.")
    print("=" * 80)

    print("\nLoading dataset...")
    df = load_energy_efficiency_dataset()
    print(f"Raw data shape: {df.shape}")

    X, y, numeric_cols = prepare_energy_features(df, TARGET_COLUMN)
    print(f"Prepared X shape: {X.shape}")
    print(f"Target summary: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}, std={y.std():.3f}")

    # Keep original indices to prove all models use the same fixed test set.
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pd.DataFrame({"test_index": idx_test}).to_csv(OUTPUT_DIR / "results" / "test_indices.csv", index=False)
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"Saved fixed test indices to: {OUTPUT_DIR / 'results' / 'test_indices.csv'}")

    models = get_models()
    results: List[BenchmarkResult] = []
    predictions: Dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        try:
            result, pred = fit_predict_one_model(
                model_name=model_name,
                model=model,
                X_train_raw=X_train,
                X_test_raw=X_test,
                y_train=y_train,
                y_test=y_test,
                numeric_cols=numeric_cols,
            )
            results.append(result)
            predictions[model_name] = pred
            print(
                f"{model_name}: "
                f"MAE={result.mae:.4f}, "
                f"RMSE={result.rmse:.4f}, "
                f"R2={result.r2:.4f}, "
                f"time={result.train_time_sec:.2f}s"
            )
        except Exception as exc:
            print(f"[Warning] {model_name} failed: {exc}")

    if not results:
        raise RuntimeError("No model finished successfully. Check package installation and error messages above.")

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("r2", ascending=False)
    results_path = OUTPUT_DIR / "results" / "model_results.csv"
    results_df.to_csv(results_path, index=False)

    pred_df = pd.DataFrame({"test_index": idx_test, "y_true": y_test})
    for model_name, pred in predictions.items():
        pred_df[f"pred_{model_name}"] = pred
        pred_df[f"residual_{model_name}"] = pred - y_test
    pred_path = OUTPUT_DIR / "results" / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print("\n" + "=" * 80)
    print("Final model results, sorted by R2")
    print(results_df[["model", "mae", "rmse", "r2", "train_time_sec"]].to_string(index=False))
    print("=" * 80)

    plot_results(results_df, y_test, predictions, OUTPUT_DIR)

    print(f"\nSaved results to: {results_path}")
    print(f"Saved test predictions to: {pred_path}")
    print(f"Saved figures to: {OUTPUT_DIR / 'figures'}")
    print("\nMost useful figures for your paper:")
    print("1. 01_r2_sorted.png")
    print("2. 04_predicted_vs_true_grid.png")
    print("3. 05_sorted_test_predictions.png")
    print("4. 06_residual_boxplot.png")


if __name__ == "__main__":
    run_benchmark()

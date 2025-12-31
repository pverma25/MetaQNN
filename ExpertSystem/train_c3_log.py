"""
C3 training script for MetaQNN-Advisor.

- Loads merged_for_c3.csv
- Uses dataset meta-features + config features as inputs (X)
- Uses val_accuracy, val_auc, train_time_sec, stability as targets (Y)
- Performs Leave-One-Dataset-Out (LODO) evaluation based on dataset_name
- Prints detailed per-dataset logs for clearer understanding
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


INPUT_CSV = "merged_for_c3.csv"

TARGET_COLS = ["val_accuracy", "val_auc", "train_time_sec", "stability"]
ID_COLS = ["dataset_name", "dataset_path", "target_col"]

LAMBDA_TIME = 0.001
GAMMA_STABILITY = 0.05
TOP_K = 3


def compute_utility(acc, time_sec, stability,
                    lam=LAMBDA_TIME, gamma=GAMMA_STABILITY):
    return acc - lam * time_sec + gamma * stability


def build_feature_matrix(df: pd.DataFrame):
    """
    Construct the feature matrix X from a DataFrame by:
    - dropping ID columns
    - dropping target columns
    - converting string (object) columns to categorical for LightGBM
    """
    cols_to_drop = [c for c in ID_COLS if c in df.columns] + TARGET_COLS
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    X = df[feature_cols].copy()

    # Convert object/string columns to categorical for LightGBM
    obj_cols = X.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        X[c] = X[c].astype("category")

    return X, feature_cols


def train_c3_models(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Train one LightGBM regressor per target (multitask via separate models).
    """
    models = {}
    for target in TARGET_COLS:
        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(X_train, y_train[target])
        models[target] = model
    return models


def predict_metrics(models, X_test: pd.DataFrame):
    """
    Use trained models to predict each target metric for X_test.
    Returns dict: target_name -> np.array of predictions.
    """
    preds = {}
    for target, model in models.items():
        preds[target] = model.predict(X_test)
    return preds


def main():
    # ---------------- Load data ----------------
    df = pd.read_csv(INPUT_CSV)
    print(f"\n Loaded {df.shape[0]} rows from {INPUT_CSV}")
    print("   Columns:", df.columns.tolist())

    for col in TARGET_COLS:
        if col not in df.columns:
            raise ValueError(f"Target column {col} not found in CSV!")

    if "dataset_name" not in df.columns:
        raise ValueError("dataset_name column is required for LODO evaluation.")

    datasets = df["dataset_name"].unique()
    n_datasets = len(datasets)
    print(f"\n Found {n_datasets} unique datasets for LODO evaluation.\n")

    # ---------------- Metrics storage ----------------
    top1_hits = []
    topk_hits = []
    regrets = []
    regrets_acc = []

    # ---------------- LODO loop ----------------
    for i, d in enumerate(datasets, start=1):
        print("=" * 80)
        print(f" LODO iteration {i}/{n_datasets}")
        print(f" Using dataset '{d}' as TEST set (completely excluded from training).")

        # Split into train and test based on dataset_name
        train_df = df[df["dataset_name"] != d].reset_index(drop=True)
        test_df = df[df["dataset_name"] == d].reset_index(drop=True)

        train_datasets = train_df["dataset_name"].unique()
        test_datasets = test_df["dataset_name"].unique()

        print(f"    Training on {len(train_datasets)} datasets "
              f"({len(train_df)} rows).")
        print(f"    Testing  on 1 dataset '{d}' "
              f"({len(test_df)} rows).")

        # Build feature matrices
        X_train, feature_cols = build_feature_matrix(train_df)
        X_test, _ = build_feature_matrix(test_df)
        y_train = train_df[TARGET_COLS]

        # Train C3 models on all other datasets
        print("    Training C3 models on training split...")
        models = train_c3_models(X_train, y_train)

        # Predict metrics on the held-out dataset
        print("    Predicting metrics for held-out test dataset...")
        preds = predict_metrics(models, X_test)

        # True metrics for test dataset
        true_acc = test_df["val_accuracy"].values
        true_time = test_df["train_time_sec"].values
        true_stab = test_df["stability"].values

        # Predicted metrics
        pred_acc = preds["val_accuracy"]
        pred_time = preds["train_time_sec"]
        pred_stab = preds["stability"]

        # Compute true and predicted utilities
        true_U = compute_utility(true_acc, true_time, true_stab)
        pred_U = compute_utility(pred_acc, pred_time, pred_stab)

        # Best indices
        idx_true_best = int(np.argmax(true_U))
        idx_pred_sorted = np.argsort(pred_U)[::-1]
        idx_pred_best = int(idx_pred_sorted[0])

        # Top-1 and Top-k hits
        hit1 = int(idx_true_best == idx_pred_best)
        top1_hits.append(hit1)

        topk = int(idx_true_best in idx_pred_sorted[:TOP_K])
        topk_hits.append(topk)

        # Regrets
        regret_U = float(true_U[idx_true_best] - true_U[idx_pred_best])
        regret_acc = float(true_acc[idx_true_best] - true_acc[idx_pred_best])

        regrets.append(regret_U)
        regrets_acc.append(regret_acc)

        # -------- Pretty per-dataset log --------
        print("\n    Per-dataset evaluation summary")
        print(f"   ─────────────────────────────────────────────")
        print(f"    Test dataset        : {d}")
        print(f"    True best index     : {idx_true_best}")
        print(f"    Predicted best index: {idx_pred_best}")
        print(f"    Top-1 hit           : {hit1} "
              f"({'✔' if hit1 == 1 else '✘'})")
        print(f"    Top-{TOP_K} hit        : {topk} "
              f"({'✔' if topk == 1 else '✘'})")
        print(f"    Regret (utility)    : {regret_U:.4f}")
        print(f"    Regret (accuracy)   : {regret_acc:.4f}")
        print("=" * 80 + "\n")

    # ---------------- Overall summary ----------------
    top1_rate = float(np.mean(top1_hits))
    topk_rate = float(np.mean(topk_hits))
    avg_regret_U = float(np.mean(regrets))
    avg_regret_acc = float(np.mean(regrets_acc))

    print("\n LODO EVALUATION COMPLETE")
    print("==============================================")
    print(f" Top-1 hit rate      : {top1_rate:.3f}")
    print(f" Top-{TOP_K} hit rate   : {topk_rate:.3f}")
    print(f" Avg regret (utility): {avg_regret_U:.4f}")
    print(f" Avg regret (acc)    : {avg_regret_acc:.4f}")
    print("==============================================\n")

    # Save per-dataset summary
    summary = pd.DataFrame({
        "dataset_name": datasets,
        "top1_hit": top1_hits,
        f"top{TOP_K}_hit": topk_hits,
        "regret_utility": regrets,
        "regret_accuracy": regrets_acc,
    })
    summary.to_csv("c3_lodo_summary.csv", index=False)
    print("💾 Saved per-dataset LODO metrics to c3_lodo_summary.csv")


if __name__ == "__main__":
    main()


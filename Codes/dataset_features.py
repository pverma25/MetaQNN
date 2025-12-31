from typing import Dict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from .config import PCA_EXPLAINED_VAR, BASELINE_CV_FOLDS, RANDOM_STATE


def compute_dataset_features(
    X_encoded: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Compute dataset-level features:
    - n_samples, n_features
    - correlation_mean
    - imbalance_ratio
    - intrinsic_dimension (PCA 95% variance)
    - feature_redundancy
    - baseline_linear_accuracy
    - separability (proxy == baseline accuracy)
    - baseline_acc_std (CV std)
    """

    n_samples, n_features = X_encoded.shape

    # Correlation mean (absolute)
    if n_features > 1:
        corr_matrix = np.corrcoef(X_encoded, rowvar=False)
        mask = ~np.eye(n_features, dtype=bool)
        corr_vals = np.abs(corr_matrix[mask])
        correlation_mean = float(np.nanmean(corr_vals)) if corr_vals.size > 0 else 0.0
    else:
        correlation_mean = 0.0

    # Imbalance ratio
    unique, counts = np.unique(y, return_counts=True)
    if len(counts) > 1:
        imbalance_ratio = float(counts.max() / counts.min())
    else:
        imbalance_ratio = 1.0

    # PCA intrinsic dimension
    if n_features > 1:
        pca = PCA()
        pca.fit(X_encoded)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = int(np.searchsorted(cum_var, PCA_EXPLAINED_VAR) + 1)
        intrinsic_dim = min(intrinsic_dim, n_features)
    else:
        intrinsic_dim = 1

    feature_redundancy = 1.0 - intrinsic_dim / float(n_features)

    # Baseline linear model (logistic regression)
    n_classes = len(np.unique(y))
    n_splits = min(BASELINE_CV_FOLDS, n_classes) if n_classes > 1 else 2
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    base_clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="auto",
    )
    scores = cross_val_score(base_clf, X_encoded, y, cv=skf, scoring="accuracy")
    baseline_linear_accuracy = float(scores.mean())
    baseline_acc_std = float(scores.std())

    separability = baseline_linear_accuracy

    return {
        "n_samples": float(n_samples),
        "n_features": float(n_features),
        "correlation_mean": correlation_mean,
        "imbalance_ratio": imbalance_ratio,
        "intrinsic_dimension": float(intrinsic_dim),
        "feature_redundancy": feature_redundancy,
        "baseline_linear_accuracy": baseline_linear_accuracy,
        "separability": separability,
        "baseline_acc_std": baseline_acc_std,
    }

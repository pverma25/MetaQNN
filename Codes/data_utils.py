from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(
    path: str,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset from CSV."""
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in dataset columns")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    val_split: float,
    test_split: float,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split for classification."""
    # First split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, stratify=y, random_state=random_state
    )
    # Then split train/val
    val_ratio = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )
    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


class TabularPreprocessor:
    """
    Generic preprocessing:
    - Detect categorical vs numeric columns
    - One-hot encode categoricals
    - Standard-scale numerics
    """

    def __init__(self):
        self.categorical_cols: List[str] = []
        self.numeric_cols: List[str] = []
        self.ohe: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame):
        # Detect column types
        self.categorical_cols = [
            c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")
        ]
        self.numeric_cols = [c for c in X.columns if c not in self.categorical_cols]

        # Fit encoders
        if self.categorical_cols:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.ohe.fit(X[self.categorical_cols])

        if self.numeric_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.numeric_cols])

        # Construct feature names
        feature_names = []
        if self.numeric_cols:
            feature_names.extend(self.numeric_cols)
        if self.categorical_cols and self.ohe is not None:
            ohe_feature_names = self.ohe.get_feature_names_out(self.categorical_cols)
            feature_names.extend(list(ohe_feature_names))
        self.feature_names_ = feature_names

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.numeric_cols and self.scaler is not None:
            parts.append(self.scaler.transform(X[self.numeric_cols]))
        if self.categorical_cols and self.ohe is not None:
            parts.append(self.ohe.transform(X[self.categorical_cols]))
        if not parts:
            raise ValueError("No features found to transform.")
        X_arr = np.concatenate(parts, axis=1)
        return X_arr

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

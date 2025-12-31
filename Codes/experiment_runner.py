from typing import Dict, List
import os
import numpy as np
import pandas as pd
import torch

from .config import (
    get_all_qnn_configs,
    qnn_config_to_dict,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_STATE,
    DEFAULT_N_QUBITS_MAX,
    MAX_EPOCHS,
)
from .data_utils import load_dataset, split_train_val_test, TabularPreprocessor
from .dataset_features import compute_dataset_features
from .qnn_models import (
    build_qnode,
    resolve_layers_for_combo,
    init_weight_shape,
    QuantumClassifier,
)
from .training import create_dataloaders, train_qnn, evaluate_model


def run_experiments_for_dataset(
    data_path: str,
    target_col: str,
    output_path: str,
):
    """
    Orchestrator for a single dataset.
    """
    X_raw, y_raw = load_dataset(data_path, target_col)
    (
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y_train,
        y_val,
        y_test,
    ) = split_train_val_test(
        X_raw,
        y_raw,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        random_state=RANDOM_STATE,
    )

    preproc = TabularPreprocessor()
    X_train_enc = preproc.fit_transform(X_train_raw)
    X_val_enc = preproc.transform(X_val_raw)
    X_test_enc = preproc.transform(X_test_raw)

    X_all_enc = np.concatenate([X_train_enc, X_val_enc, X_test_enc], axis=0)
    y_all = np.concatenate([y_train.values, y_val.values, y_test.values], axis=0)
    dataset_features = compute_dataset_features(X_all_enc, y_all)

    n_qubits = min(X_train_enc.shape[1], DEFAULT_N_QUBITS_MAX)

    if X_train_enc.shape[1] > n_qubits:
        X_train_q = X_train_enc[:, :n_qubits]
        X_val_q = X_val_enc[:, :n_qubits]
        X_test_q = X_test_enc[:, :n_qubits]
    else:
        X_train_q = X_train_enc
        X_val_q = X_val_enc
        X_test_q = X_test_enc

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_q,
        y_train.values,
        X_val_q,
        y_val.values,
        X_test_q,
        y_test.values,
    )

    #configs = get_all_qnn_configs()
    configs = get_all_qnn_configs()[:1]  # run only first 1 config
    rows: List[Dict] = []

    for cfg in configs:
        n_layers = resolve_layers_for_combo(cfg.layer_combo)
        weight_shape = init_weight_shape(n_qubits, n_layers)
        circuit = build_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz_type=cfg.ansatz_type,
            feature_map=cfg.feature_map,
        )
        model = QuantumClassifier(circuit, weight_shape)

        val_metrics, train_metrics, train_time, epochs_run = train_qnn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_name=cfg.optimizer,
        )

        test_metrics = evaluate_model(model, test_loader)

        row: Dict = {}
        row["dataset_path"] = data_path
        row["target_col"] = target_col

        row.update(dataset_features)

        row.update(qnn_config_to_dict(cfg))
        row["n_qubits"] = n_qubits
        row["n_layers"] = n_layers
        row["n_params"] = int(np.prod(weight_shape))

        row["max_epochs"] = MAX_EPOCHS
        row["epochs_run"] = epochs_run
        row["train_time_sec"] = train_time

        row["train_accuracy"] = train_metrics["accuracy"]
        row["train_f1"] = train_metrics["f1"]
        row["train_auc"] = train_metrics["auc"]
        row["train_loss"] = train_metrics["loss"]

        row["val_accuracy"] = val_metrics["accuracy"]
        row["val_f1"] = val_metrics["f1"]
        row["val_auc"] = val_metrics["auc"]
        row["val_loss"] = val_metrics["loss"]

        row["test_accuracy"] = test_metrics["accuracy"]
        row["test_f1"] = test_metrics["f1"]
        row["test_auc"] = test_metrics["auc"]
        row["test_loss"] = test_metrics["loss"]

        rows.append(row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_new = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        df_old = pd.read_csv(output_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(output_path, index=False)
    print(f"Saved {len(df_new)} rows to {output_path}")

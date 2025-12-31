from dataclasses import dataclass
from typing import List, Dict

# Training hyperparameters
MAX_EPOCHS = 40
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 6
LEARNING_RATE = 1e-2
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
RANDOM_STATE = 42

# PCA threshold for intrinsic dimension
PCA_EXPLAINED_VAR = 0.95

# Number of CV folds for baseline linear model
BASELINE_CV_FOLDS = 5

# Quantum-related defaults
DEFAULT_N_QUBITS_MAX = 8  # cap to keep simulations light


@dataclass
class QNNConfig:
    layer_combo: str         # "shallow", "deep", "strong_entangling", "hybrid"
    ansatz_type: str         # "hardware_efficient", "strongly_entangling"
    feature_map: str         # "z", "zz", "pauli"
    optimizer: str           # "adam", "rmsprop"


def get_all_qnn_configs() -> List[QNNConfig]:
    layer_combos = ["shallow", "deep", "strong_entangling", "hybrid"]
    ansatz_types = ["hardware_efficient", "strongly_entangling"]
    feature_maps = ["z", "zz", "pauli"]
    optimizers = ["adam", "rmsprop"]

    configs: List[QNNConfig] = []
    for lc in layer_combos:
        for at in ansatz_types:
            for fm in feature_maps:
                for opt in optimizers:
                    configs.append(QNNConfig(
                        layer_combo=lc,
                        ansatz_type=at,
                        feature_map=fm,
                        optimizer=opt
                    ))
    return configs


def qnn_config_to_dict(cfg: QNNConfig) -> Dict[str, str]:
    return {
        "layer_combo": cfg.layer_combo,
        "ansatz_type": cfg.ansatz_type,
        "feature_map": cfg.feature_map,
        "optimizer_type": cfg.optimizer,
    }

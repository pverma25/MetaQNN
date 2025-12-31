from typing import Tuple
import pennylane as qml
import torch
import torch.nn as nn


def build_qnode(
    n_qubits: int,
    n_layers: int,
    ansatz_type: str,
    feature_map: str,
    dev_name: str = "default.qubit",
):
    """
    Build a PennyLane qnode with specified ansatz and feature map.
    Assumes binary classification: output is expectation of PauliZ on last qubit.
    """
    dev = qml.device(dev_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        # FEATURE MAPS
        if feature_map == "z":
            for i in range(n_qubits):
                qml.RZ(inputs[i], wires=i)

        elif feature_map == "zz":
            for i in range(n_qubits):
                qml.RZ(inputs[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(inputs[i] * inputs[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])

        elif feature_map == "pauli":
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
                qml.RZ(inputs[i], wires=i)
        else:
            raise ValueError(f"Unknown feature_map '{feature_map}'")

        # ANSATZ
        if ansatz_type == "hardware_efficient":
            for l in range(n_layers):
                for i in range(n_qubits):
                    th = weights[l, i]
                    qml.RX(th[0], wires=i)
                    qml.RY(th[1], wires=i)
                    qml.RZ(th[2], wires=i)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

        elif ansatz_type == "strongly_entangling":
            for l in range(n_layers):
                for i in range(n_qubits):
                    th = weights[l, i]
                    qml.RY(th[0], wires=i)
                    qml.RZ(th[1], wires=i)
                    qml.RX(th[2], wires=i)
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
        else:
            raise ValueError(f"Unknown ansatz_type '{ansatz_type}'")

        return qml.expval(qml.PauliZ(wires=n_qubits - 1))

    return circuit


def resolve_layers_for_combo(layer_combo: str) -> int:
    if layer_combo == "shallow":
        return 2
    elif layer_combo == "deep":
        return 6
    elif layer_combo == "strong_entangling":
        return 4
    elif layer_combo == "hybrid":
        return 4
    else:
        raise ValueError(f"Unknown layer_combo '{layer_combo}'")


def init_weight_shape(n_qubits: int, n_layers: int) -> Tuple[int, int, int]:
    return (n_layers, n_qubits, 3)


class QuantumClassifier(nn.Module):
    """A simple PyTorch module wrapping a PennyLane QNN for binary classification."""

    def __init__(self, circuit, weight_shape: Tuple[int, int, int]):
        super().__init__()
        self.circuit = circuit
        self.weight_shape = weight_shape
        init_weights = 0.01 * torch.randn(*weight_shape)
        self.weights = nn.Parameter(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expectations = []
        for i in range(x.shape[0]):
            out = self.circuit(x[i], self.weights)
            expectations.append(out)
        expvals = torch.stack(expectations)
        probs1 = (1.0 - expvals) / 2.0
        probs0 = 1.0 - probs1
        probs = torch.stack([probs0, probs1], dim=1)
        return probs

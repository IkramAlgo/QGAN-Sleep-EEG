# qgan/models.py
# Quantum Generator and Discriminator circuits using PennyLane + PyTorch

import torch
import pennylane as qml

from qgan.config import N_LAYERS, WEIGHT_INIT_STD


def _run_circuit(circuit, x, weights):
    out = circuit(x, weights)
    return torch.stack(out) if isinstance(out, (list, tuple)) else out


class GeneratorQuantumCircuit(torch.nn.Module):
    # takes random noise, outputs fake sleep data in range [-1, 1]

    def __init__(self, n_qubits=4, n_layers=N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device   = qml.device("default.qubit", wires=n_qubits)
        self.weights  = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for w in range(n_qubits):
                qml.RX(inputs[w], wires=w)
            for l in range(n_layers):
                for w in range(n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % n_qubits])
                for w in range(n_qubits):
                    qml.RY(weights[l, w], wires=w)
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x):
        if x.dim() == 1:
            return _run_circuit(self.circuit, x, self.weights)
        return torch.stack([
            _run_circuit(self.circuit, xi, self.weights) for xi in x
        ])


class DiscriminatorQuantumCircuit(torch.nn.Module):
    # takes real or fake data, outputs judgment values in range [-1, 1]

    def __init__(self, n_qubits=4, n_layers=N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device   = qml.device("default.qubit", wires=n_qubits)
        self.weights  = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for w in range(n_qubits):
                qml.RY(inputs[w], wires=w)
            for l in range(n_layers):
                for w in range(n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % n_qubits])
                for w in range(n_qubits):
                    qml.RZ(weights[l, w], wires=w)
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x):
        if x.dim() == 1:
            return _run_circuit(self.circuit, x, self.weights)
        return torch.stack([
            _run_circuit(self.circuit, xi, self.weights) for xi in x
        ])
# qgan/models_ibm.py
# QGAN circuits using IBM 127-qubit noise model (Fake127QPulseV1)
# Scientifically valid QPU simulation for paper

import torch
import pennylane as qml
from qgan.config import N_LAYERS, WEIGHT_INIT_STD


def _run_circuit(circuit, x, weights):
    out = circuit(x, weights)
    return torch.stack(out) if isinstance(out, (list, tuple)) else out


def get_ibm_device(n_qubits, shots=512):
    """
    Uses Fake127QPulseV1 — real IBM 127-qubit hardware noise profile.
    This is the correct QPU simulation for the paper.
    Paper label: QPU-Sim (IBM 127-qubit, Fake127QPulseV1)
    """
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        from qiskit.providers.fake_provider import Fake127QPulseV1
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress V1 deprecation warning
            fake_backend = Fake127QPulseV1()
            noise_model  = NoiseModel.from_backend(fake_backend)

        dev   = qml.device("qiskit.aer", wires=n_qubits,
                            noise_model=noise_model, shots=shots)
        label = f"QPU-Sim (IBM 127-qubit, {shots} shots)"
        print(f"  Device: {label}")
        return dev, label

    except Exception as e:
        print(f"  Noise sim failed: {e}")
        print(f"  Falling back to noiseless CPU sim")
        dev   = qml.device("default.qubit", wires=n_qubits)
        label = "CPU-Sim (noiseless)"
        return dev, label


class GeneratorIBM(torch.nn.Module):
    def __init__(self, n_qubits=4, n_layers=N_LAYERS, shots=512):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev, self.device_label = get_ibm_device(n_qubits, shots)
        self.weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits, 2) * WEIGHT_INIT_STD
        )

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            for w in range(n_qubits):
                qml.RY(inputs[w], wires=w)
            for l in range(n_layers):
                for w in range(n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
                for w in range(n_qubits):
                    qml.RZ(weights[l, w, 0], wires=w)
                    qml.RY(weights[l, w, 1], wires=w)
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x):
        if x.dim() == 1:
            return _run_circuit(self.circuit, x, self.weights)
        return torch.stack([
            _run_circuit(self.circuit, xi, self.weights) for xi in x
        ])


class DiscriminatorIBM(torch.nn.Module):
    def __init__(self, n_qubits=4, n_layers=N_LAYERS, shots=512):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev, self.device_label = get_ibm_device(n_qubits, shots)
        self.weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits, 2) * WEIGHT_INIT_STD
        )

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            for w in range(n_qubits):
                qml.RY(inputs[w], wires=w)
            for l in range(n_layers):
                for w in range(n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
                for w in range(n_qubits):
                    qml.RZ(weights[l, w, 0], wires=w)
                    qml.RY(weights[l, w, 1], wires=w)
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x):
        if x.dim() == 1:
            return _run_circuit(self.circuit, x, self.weights)
        return torch.stack([
            _run_circuit(self.circuit, xi, self.weights) for xi in x
        ])
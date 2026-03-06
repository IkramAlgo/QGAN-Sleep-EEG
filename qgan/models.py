# Quantum Generative Adversarial Network (QGAN)
# Updated: small weight initialisation to avoid barren plateaus

import torch
import pennylane as qml


class DiscriminatorQuantumCircuit(torch.nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device   = qml.device("default.qubit", wires=self.n_qubits)

        # FIXED: small init (0.01) to avoid barren plateau vanishing gradients
        self.weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits) * 0.01
        )

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # encode real/fake data as rotation angles
            for wire in range(self.n_qubits):
                qml.RY(inputs[wire], wires=wire)
            # trainable layers with entanglement
            for layer in range(self.n_layers):
                for wire in range(self.n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])
                    qml.RZ(weights[layer, wire], wires=wire)
            return [qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, inputs):
        if inputs.dim() == 1:
            out = self.circuit(inputs, self.weights)
            return torch.stack(out) if isinstance(out, (list, tuple)) else out
        outputs = []
        for x in inputs:
            out = self.circuit(x, self.weights)
            outputs.append(torch.stack(out) if isinstance(out, (list, tuple)) else out)
        return torch.stack(outputs)


class GeneratorQuantumCircuit(torch.nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device   = qml.device("default.qubit", wires=self.n_qubits)

        # FIXED: small init (0.01) to avoid barren plateau vanishing gradients
        self.weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits) * 0.01
        )

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for wire in range(self.n_qubits):
                qml.RX(inputs[wire], wires=wire)
            for layer in range(self.n_layers):
                for wire in range(self.n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])
                    qml.RY(weights[layer, wire], wires=wire)
            return [qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, inputs):
        if inputs.dim() == 1:
            out = self.circuit(inputs, self.weights)
            return torch.stack(out) if isinstance(out, (list, tuple)) else out
        outputs = []
        for x in inputs:
            out = self.circuit(x, self.weights)
            outputs.append(torch.stack(out) if isinstance(out, (list, tuple)) else out)
        return torch.stack(outputs)
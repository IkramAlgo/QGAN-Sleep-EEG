# qgan/models.py
# Architecture 1 — Hybrid QGAN
# Quantum Generator (unchanged) + Classical Discriminator (bigger, with Dropout)
# Changes from v1:
#   - Discriminator upgraded: 16->8->1  to  32->16->1 with Dropout(0.3)
#   - Reason: bigger discriminator has more capacity to separate
#             quantum generator outputs from real EEG data

import torch
import pennylane as qml

from qgan.config import N_LAYERS, WEIGHT_INIT_STD


def _run_circuit(circuit, x, weights):
    out = circuit(x, weights)
    return torch.stack(out) if isinstance(out, (list, tuple)) else out


class GeneratorQuantumCircuit(torch.nn.Module):
    # UNCHANGED — exact same quantum generator as original
    # takes random noise, outputs fake sleep data in range [-1, 1]
    # RX encoding -> CNOT entanglement -> RY trainable weights

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


class ClassicalDiscriminator(torch.nn.Module):
    # UPDATED — bigger network with Dropout
    # Previous: Linear(input->16) -> Linear(16->8) -> Linear(8->1)
    # Now:      Linear(input->32) -> Dropout(0.3) -> Linear(32->16) -> Linear(16->1)
    # Why bigger: more capacity to separate quantum generator outputs from real data
    # Why dropout: prevents memorization, forces better generalization
    # input_dim must match n_features used in training (2, 3, or 4)

    def __init__(self, input_dim=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # .float() converts quantum generator double output to float32
        return self.net(x.float())
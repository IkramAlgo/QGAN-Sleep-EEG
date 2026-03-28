# qgan/models_noise.py
# Three models for noise experiment comparison:
#   1. Classical GAN Generator + Discriminator
#   2. QGAN Arch C Generator (6 qubits, ring CNOT, RX->CNOT->RY)
#   3. Shared Classical Discriminator (no Sigmoid — WGAN-GP)
#
# Uses backprop + default.qubit for maximum speed on local CPU.
# Arch C circuit is identical to CPU ablation study.

import torch
import torch.nn as nn
import pennylane as qml
from qgan.config import N_LAYERS, WEIGHT_INIT_STD


# ================================================================
#  DATA NOISE — inspired by QCNN Qiskit generate_dataset function
#  Adds Gaussian noise to EEG features, keeps in [-1, 1] range
# ================================================================
def add_data_noise(data, noise_level=0.1):
    """
    Add Gaussian noise to EEG feature data.
    Inspired by QCNN Qiskit tutorial generate_dataset noise injection.

    Args:
        data       : tensor of shape [N, n_features], values in [-1, 1]
        noise_level: standard deviation of noise (0.1 = 10% noise)

    Returns:
        noisy_data : same shape, clamped to [-1, 1]
    """
    noise      = torch.randn_like(data) * noise_level
    noisy_data = data + noise
    return torch.clamp(noisy_data, -1.0, 1.0)


# ================================================================
#  CLASSICAL GAN GENERATOR
#  Simple MLP: noise -> fake EEG features
#  Fast baseline for comparison
# ================================================================
class ClassicalGenerator(nn.Module):
    """
    Classical GAN generator — MLP with Tanh output.
    Input:  random noise [batch, latent_dim]
    Output: fake EEG features [batch, n_features], range [-1, 1]
    """
    def __init__(self, latent_dim=4, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, output_dim),
            nn.Tanh()   # output in [-1, 1] to match quantum circuit range
        )

    def forward(self, z):
        return self.net(z.float())


# ================================================================
#  QUANTUM GENERATOR — ARCH C
#  6 qubits | ring CNOT | RX encoding | RY trainable | 2 layers
#  Uses default.qubit + backprop for maximum local CPU speed
#  Identical circuit to CPU Arch C ablation study
# ================================================================
class GeneratorArchC(nn.Module):
    """
    Arch C quantum generator.
    Input:  noise [batch, n_features]   — n_features <= 6
    Output: fake EEG [batch, 6]         — 6 qubit measurements
    First n_features outputs used as EEG features.
    """
    def __init__(self, n_qubits=6, n_layers=N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # default.qubit + backprop = fastest option for local CPU
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # [n_layers, n_qubits] — one RY per qubit per layer
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Step 1: RX encoding — first n inputs, rest zero-padded
            for w in range(n_qubits):
                angle = inputs[w] if w < len(inputs) else torch.tensor(0.0)
                qml.RX(angle, wires=w)

            # Step 2: Ring CNOT + RY trainable, repeated n_layers times
            for l in range(n_layers):
                # Ring: 0->1->2->3->4->5->0
                for w in range(n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % n_qubits])
                # Trainable RY
                for w in range(n_qubits):
                    qml.RY(weights[l, w], wires=w)

            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x):
        x = x.float()

        def _run(xi):
            # Pad input from n_features to n_qubits
            if xi.shape[0] < self.n_qubits:
                pad = torch.zeros(self.n_qubits - xi.shape[0])
                xi  = torch.cat([xi, pad])
            out = self.circuit(xi, self.weights)
            return torch.stack(out) if isinstance(out, (list, tuple)) else out

        if x.dim() == 1:
            return _run(x)
        return torch.stack([_run(xi) for xi in x])


# ================================================================
#  SHARED CLASSICAL DISCRIMINATOR
#  Used by ALL three models for fair comparison.
#  input_dim = N_QUBITS (6) because quantum generator outputs 6 values.
#  Classical GAN real/fake data padded to 6 dims before passing in.
#  No Sigmoid — WGAN-GP needs raw scores.
#  Classical GAN uses BCE so sigmoid applied externally in that case.
# ================================================================
class ClassicalDiscriminator(nn.Module):
    """
    Shared discriminator for all three models.
    Input:  [batch, input_dim]
    Output: [batch, 1] raw score (no sigmoid)
    """
    def __init__(self, input_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            # NO Sigmoid here — applied externally for BCE, not needed for WGAN-GP
        )

    def forward(self, x):
        return self.net(x.float())
# qgan/models_journal.py
# Four backend conditions for journal quantum error correction study.
#
# CONDITION 1: Simulator          — default.qubit, noiseless, backprop
# CONDITION 2: Simulator+DataNoise— default.qubit + Gaussian input noise
# CONDITION 3: QPU-Sim            — FakeNairobi hardware noise, no mitigation
# CONDITION 4: QPU-Sim+ZNE        — FakeNairobi + Zero-Noise Extrapolation
#
# Architecture: Arch C (6 qubits, ring CNOT, RX->CNOT->RY, 2 layers)
# Same circuit across ALL 4 conditions — only the device/mitigation changes.
# Classical Discriminator shared across all conditions for fair comparison.
#
# References:
#   ZNE: Temme et al. (2017), PennyLane mitigate_with_zne
#   FakeNairobi: IBM 7-qubit noise model (fits in RAM, realistic noise)

import os
import warnings
import torch
import torch.nn as nn
import pennylane as qml
from qgan.config import N_LAYERS, WEIGHT_INIT_STD

# ================================================================
#  BACKEND CONDITION LABELS — used as keys throughout training
# ================================================================
CONDITION_SIMULATOR    = "simulator"
CONDITION_DATA_NOISE   = "simulator_datanoise"
CONDITION_QPU_SIM      = "qpu_noiseless"       # QPU noise model, no QEC
CONDITION_QPU_ZNE      = "qpu_noise_zne"        # QPU noise model + ZNE

ALL_CONDITIONS = [
    CONDITION_SIMULATOR,
    CONDITION_DATA_NOISE,
    CONDITION_QPU_SIM,
    CONDITION_QPU_ZNE,
]

CONDITION_LABELS = {
    CONDITION_SIMULATOR:  "Simulator (noiseless)",
    CONDITION_DATA_NOISE: "Simulator + DataNoise",
    CONDITION_QPU_SIM:    "QPU-Sim (no QEC)",
    CONDITION_QPU_ZNE:    "QPU-Sim + ZNE",
}


# ================================================================
#  QUANTUM DEVICE FACTORY
#  Returns the correct PennyLane device for each condition.
#  FakeNairobi = 7-qubit IBM noise model — small enough for local RAM.
#  On ARC with real QPU, replace with actual ibm_torino backend.
# ================================================================
def _get_device(condition: str, n_qubits: int = 6):
    """
    Build the correct PennyLane device for the given condition.

    Conditions 1+2 use default.qubit (noiseless CPU).
    Conditions 3+4 use FakeNairobi noise model (7-qubit IBM device).
    ZNE is applied as a transform wrapper, not at device level.
    """
    if condition in (CONDITION_SIMULATOR, CONDITION_DATA_NOISE):
        # Fast noiseless simulation — backprop compatible
        return qml.device("default.qubit", wires=n_qubits), "backprop"

    # Hardware noise model — try FakeNairobi (7-qubit, fits in RAM)
    # Fall back to noiseless if Qiskit noise models unavailable
    try:
        from qiskit_aer import AerSimulator
        from qiskit.providers.fake_provider import FakeNairobi
        noise_model = FakeNairobi()
        dev = qml.device(
            "qiskit.aer",
            wires=n_qubits,
            backend=AerSimulator(noise_model=noise_model),
            shots=128,
        )
        return dev, "parameter-shift"
    except Exception as e:
        warnings.warn(
            f"FakeNairobi unavailable ({e}). "
            f"Falling back to noiseless default.qubit for {condition}. "
            f"Results will match Condition 1, not QPU noise."
        )
        return qml.device("default.qubit", wires=n_qubits), "backprop"


# ================================================================
#  ARCH C QUANTUM GENERATOR
#  6 qubits | ring CNOT | RX encoding | RY trainable | 2 layers
#  Identical circuit for all 4 conditions — only device changes.
# ================================================================
class GeneratorJournal(nn.Module):
    """
    Arch C quantum generator for all 4 journal backend conditions.

    Args:
        condition  : one of ALL_CONDITIONS
        n_qubits   : 6 (Arch C standard)
        n_layers   : 2 (Arch C standard)
        n_features : number of EEG features (2, 3, or 4)
    """
    def __init__(self,
                 condition: str = CONDITION_SIMULATOR,
                 n_qubits: int = 6,
                 n_layers: int = N_LAYERS,
                 n_features: int = 4):
        super().__init__()
        self.condition  = condition
        self.n_qubits   = n_qubits
        self.n_layers   = n_layers
        self.n_features = n_features

        dev, diff_method = _get_device(condition, n_qubits)
        self.dev_name    = str(dev)

        # ZNE scale factors — used only for CONDITION_QPU_ZNE
        self.zne_scale_factors = [1, 3, 5]

        # Trainable weights: [n_layers, n_qubits]
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )

        # Build base circuit
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def base_circuit(inputs, weights):
            # RX encoding — first n_features qubits get input, rest zero-padded
            for w in range(n_qubits):
                angle = inputs[w] if w < len(inputs) else torch.tensor(0.0)
                qml.RX(angle, wires=w)

            # Ring CNOT + RY, n_layers times
            for layer in range(n_layers):
                for w in range(n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % n_qubits])
                for w in range(n_qubits):
                    qml.RY(weights[layer, w], wires=w)

            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self._base_circuit = base_circuit

        # ZNE-wrapped circuit (only built for QPU+ZNE condition)
        if condition == CONDITION_QPU_ZNE:
            self._circuit = self._build_zne_circuit(base_circuit, dev, diff_method)
        else:
            self._circuit = base_circuit

    def _build_zne_circuit(self, base_circuit, dev, diff_method):
        """
        Wrap the base circuit with Zero-Noise Extrapolation.
        Runs circuit at noise_factors [1, 3, 5] and extrapolates to zero noise.
        Uses polynomial extrapolation (order 2).

        This is the key novel contribution for the journal paper:
        asking whether ZNE improves QGAN biomedical generation quality.
        """
        try:
            from pennylane.transforms import mitigate_with_zne, poly_extrapolate

            mitigated = mitigate_with_zne(
                base_circuit,
                scale_factors=self.zne_scale_factors,
                extrapolate=poly_extrapolate,
                extrapolate_kwargs={"order": 2},
            )
            return mitigated
        except ImportError:
            warnings.warn(
                "PennyLane ZNE transforms not available. "
                "Running QPU+ZNE condition WITHOUT ZNE mitigation. "
                "Upgrade PennyLane: pip install pennylane --upgrade"
            )
            return base_circuit

    def _run_single(self, xi: torch.Tensor) -> torch.Tensor:
        """Run circuit for a single input vector [n_qubits]."""
        # Pad input to n_qubits
        if xi.shape[0] < self.n_qubits:
            pad = torch.zeros(self.n_qubits - xi.shape[0])
            xi  = torch.cat([xi, pad])

        out = self._circuit(xi, self.weights)
        return torch.stack(out) if isinstance(out, (list, tuple)) else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_features] noise input
        Returns:
            [batch, n_qubits] quantum measurements
        """
        x = x.float()
        if x.dim() == 1:
            return self._run_single(x)
        return torch.stack([self._run_single(xi) for xi in x])


# ================================================================
#  CLASSICAL DISCRIMINATOR
#  Shared across all 4 conditions for fair comparison.
#  input_dim = n_qubits = 6 (real data is zero-padded to match).
#  No Sigmoid — WGAN-GP needs raw scores.
# ================================================================
class ClassicalDiscriminator(nn.Module):
    """
    Shared discriminator for all journal backend conditions.
    Input:  [batch, input_dim]  — real or fake EEG features (padded to 6)
    Output: [batch, 1]          — raw Wasserstein score (no sigmoid)
    """
    def __init__(self, input_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            # NO Sigmoid — WGAN-GP uses raw scores, threshold = 0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


# ================================================================
#  MODEL FACTORY
#  Convenience function used by train_journal.py
# ================================================================
def build_models(condition: str,
                 n_qubits: int = 6,
                 n_features: int = 4) -> tuple:
    """
    Build (generator, discriminator) pair for a given backend condition.

    Returns:
        generator     : GeneratorJournal
        discriminator : ClassicalDiscriminator
    """
    gen  = GeneratorJournal(
        condition=condition,
        n_qubits=n_qubits,
        n_layers=N_LAYERS,
        n_features=n_features,
    )
    disc = ClassicalDiscriminator(input_dim=n_qubits)
    return gen, disc
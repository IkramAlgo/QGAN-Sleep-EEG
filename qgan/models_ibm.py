# qgan/models_ibm.py
# QGAN Arch C — 6 qubits, ring CNOT, RX->CNOT->RY
# ARC VERSION: AerSimulator + custom depolarizing noise (no real QPU)
# Credentials file is NOT required — all IBM QPU logic removed.

import torch
import warnings
import pennylane as qml
from qgan.config import N_LAYERS, WEIGHT_INIT_STD


# ================================================================
#  DEVICE FACTORY
#  Always returns AerSimulator with custom depolarizing noise.
#  use_real_qpu parameter is kept for API compatibility but ignored.
# ================================================================

# --- Noise parameters (tune as needed) ---
DEPOL_1Q = 1e-3   # single-qubit depolarizing error rate
DEPOL_2Q = 5e-3   # two-qubit (CNOT) depolarizing error rate
T1       = 50e-6  # T1 relaxation time  (seconds)
T2       = 70e-6  # T2 dephasing time   (seconds)
GATE_1Q  = 50e-9  # single-qubit gate time (seconds)
GATE_2Q  = 300e-9 # two-qubit gate time    (seconds)


def _build_noise_model(n_qubits):
    """
    Build a custom depolarizing + thermal relaxation noise model.
    Applied to every qubit and every gate in the circuit.
    """
    from qiskit_aer.noise import (
        NoiseModel, depolarizing_error, thermal_relaxation_error
    )

    noise_model = NoiseModel()

    for qubit in range(n_qubits):
        # --- single-qubit gate noise (rx, ry, u1, u2, u3) ---
        dep_1q    = depolarizing_error(DEPOL_1Q, 1)
        therm_1q  = thermal_relaxation_error(T1, T2, GATE_1Q)
        combined_1q = dep_1q.compose(therm_1q)
        noise_model.add_quantum_error(
            combined_1q,
            ["rx", "ry", "rz", "h", "u1", "u2", "u3"],
            [qubit]
        )

    # --- two-qubit gate noise (cx / CNOT) ---
    dep_2q = depolarizing_error(DEPOL_2Q, 2)
    for ctrl in range(n_qubits):
        tgt = (ctrl + 1) % n_qubits   # ring topology
        therm_ctrl = thermal_relaxation_error(T1, T2, GATE_2Q)
        therm_tgt  = thermal_relaxation_error(T1, T2, GATE_2Q)
        combined_2q = dep_2q.expand(therm_ctrl.tensor(therm_tgt))
        noise_model.add_quantum_error(combined_2q, ["cx"], [ctrl, tgt])

    return noise_model


def get_ibm_device(n_qubits, shots=1024, use_real_qpu=False):
    """
    Returns (PennyLane device, label string).

    ARC mode: always AerSimulator + custom depolarizing noise.
    use_real_qpu is accepted for API compatibility but has no effect.
    """
    if use_real_qpu:
        print("  NOTE: use_real_qpu=True ignored — ARC version uses "
              "AerSimulator + custom depolarizing noise.")

    try:
        from qiskit_aer import AerSimulator

        noise_model = _build_noise_model(n_qubits)

        # AerSimulator statevector backend with noise injection
        backend = AerSimulator(noise_model=noise_model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dev = qml.device(
                "qiskit.aer",
                wires=n_qubits,
                backend=backend,
                shots=shots,
            )

        label = (
            f"AerSimulator+DepolarNoise "
            f"(1q={DEPOL_1Q:.0e}, 2q={DEPOL_2Q:.0e}, shots={shots})"
        )
        print(f"  Device : {label}")
        return dev, label

    except Exception as e:
        # Graceful fallback so the job does not crash on ARC
        print(f"  WARNING: AerSimulator failed ({e})")
        print(f"  Falling back to PennyLane default.qubit (noiseless).")
        dev   = qml.device("default.qubit", wires=n_qubits)
        label = "default.qubit (noiseless fallback)"
        return dev, label


# ================================================================
#  ARCH C GENERATOR
#  6 qubits | ring CNOT | RX encoding | RY trainable | 2 layers
#  diff_method="parameter-shift" — compatible with all Aer backends
# ================================================================
class GeneratorArchC(torch.nn.Module):

    def __init__(self, n_qubits=6, n_layers=N_LAYERS,
                 shots=1024, use_real_qpu=False):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.dev, self.device_label = get_ibm_device(
            n_qubits, shots, use_real_qpu
        )

        # [n_layers, n_qubits] — one RY per qubit per layer
        self.weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            # Step 1 — RX encoding
            # First n_features inputs = EEG features; extra qubits get 0
            for w in range(n_qubits):
                angle = inputs[w] if w < len(inputs) else torch.tensor(0.0)
                qml.RX(angle, wires=w)

            # Step 2 — Layers: ring CNOT then RY trainable weights
            for l in range(n_layers):
                # Ring CNOT: 0->1->2->3->4->5->0
                for w in range(n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % n_qubits])
                # RY trainable
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
#  CLASSICAL DISCRIMINATOR
#  input_dim=6 because generator outputs 6 qubit measurements
#  No Sigmoid — WGAN-GP needs raw scores
# ================================================================
class ClassicalDiscriminator(torch.nn.Module):

    def __init__(self, input_dim=6):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(16, 1),
            # NO Sigmoid — WGAN-GP raw scores
        )

    def forward(self, x):
        return self.net(x.float())
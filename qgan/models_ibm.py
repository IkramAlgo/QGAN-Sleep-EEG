# qgan/models_ibm.py
# QGAN Arch C — 6 qubits, ring CNOT, RX->CNOT->RY
#
# NOISE STRATEGY (3 layers):
#   Layer 1 — DATA-LEVEL NOISE   : Gaussian noise injected into RX encoding
#                                   angles before they enter the circuit.
#                                   Simulates EEG sensor noise + ADC jitter.
#   Layer 2 — CIRCUIT-LEVEL NOISE: Explicit Pauli X/Y/Z errors inserted after
#                                   every gate inside the QNode via PennyLane's
#                                   qml.PauliError (no Qiskit required).
#   Layer 3 — HARDWARE NOISE     : AerSimulator + depolarizing + thermal
#                                   relaxation (used when available; graceful
#                                   fallback to default.qubit if Aer fails).
#
# This matches the pattern:
#   if images[-1][i] == 0:
#       images[-1][i] = algorithm_globals.random.uniform(0, np.pi/4)
# …but generalised to continuous Gaussian corruption on all angles,
# plus explicit mid-circuit Pauli noise channels.

import torch
import warnings
import numpy as np
import pennylane as qml
from qgan.config import N_LAYERS, WEIGHT_INIT_STD


# ================================================================
#  NOISE PARAMETERS  — tune all three layers independently
# ================================================================

# --- Layer 1: Data-level encoding noise (Gaussian, radians) ---
ENCODING_NOISE_STD = 0.15        # σ for RX angle corruption (~8.6° std dev)
ZERO_ANGLE_NOISE   = np.pi / 4   # max uniform noise when angle ≈ 0
ZERO_THRESH        = 1e-3        # threshold to treat an angle as "zero"

# --- Layer 2: Circuit-level Pauli noise ---
PAULI_ERROR_PROB_1Q = 0.01       # error probability after each 1-qubit gate
PAULI_ERROR_PROB_2Q = 0.03       # error probability after each CNOT

# --- Layer 3: AerSimulator hardware noise ---
DEPOL_1Q = 1e-3
DEPOL_2Q = 5e-3
T1       = 50e-6
T2       = 70e-6
GATE_1Q  = 50e-9
GATE_2Q  = 300e-9


# ================================================================
#  LAYER 1 HELPER — DATA-LEVEL NOISE
#  Mirrors the example pattern:
#    if images[-1][i] == 0:
#        images[-1][i] = uniform(0, pi/4)
#  Extended: all non-zero angles also get Gaussian jitter.
# ================================================================
def inject_encoding_noise(angles: torch.Tensor, training: bool) -> torch.Tensor:
    """
    Corrupt RX encoding angles to simulate EEG sensor + ADC noise.

    Rules (applied per element):
      • If |angle| < ZERO_THRESH  → replace with Uniform(0, π/4)
        (exact replica of the example pattern for zero-valued features)
      • Otherwise                 → add Gaussian(0, ENCODING_NOISE_STD)

    Only active during training so evaluation is deterministic.
    """
    if not training:
        return angles

    noisy = angles.clone()
    flat  = noisy.view(-1)

    for i in range(flat.shape[0]):
        val = flat[i].item()
        if abs(val) < ZERO_THRESH:
            # Zero-angle noise: uniform in [0, π/4]  (same as example)
            flat[i] = float(np.random.uniform(0, ZERO_ANGLE_NOISE))
        else:
            # Non-zero angle: Gaussian jitter
            flat[i] = val + float(np.random.normal(0, ENCODING_NOISE_STD))

    return noisy.view(angles.shape)


# ================================================================
#  LAYER 3 HELPER — AERSIMULATOR NOISE MODEL
#  Fixed: thermal relaxation on cx uses expand(2) to produce a
#  proper 2-qubit error — avoids the "1-qubit error on 2-qubit
#  instruction" crash that caused the fallback in earlier versions.
# ================================================================
def _build_noise_model(n_qubits):
    from qiskit_aer.noise import (
        NoiseModel, depolarizing_error, thermal_relaxation_error
    )
    noise_model = NoiseModel()

    # Single-qubit gate noise (compose = depolarizing then thermal)
    for qubit in range(n_qubits):
        dep_1q      = depolarizing_error(DEPOL_1Q, 1)
        therm_1q    = thermal_relaxation_error(T1, T2, GATE_1Q)
        combined_1q = dep_1q.compose(therm_1q)
        noise_model.add_quantum_error(
            combined_1q,
            ["rx", "ry", "rz", "h", "u1", "u2", "u3"],
            [qubit]
        )

    # 2-qubit depolarizing on each ring pair
    dep_2q = depolarizing_error(DEPOL_2Q, 2)
    for ctrl in range(n_qubits):
        tgt = (ctrl + 1) % n_qubits
        noise_model.add_quantum_error(dep_2q, ["cx"], [ctrl, tgt])

    # Thermal relaxation on cx — expand(2) makes it a valid 2-qubit error ✓
    therm_2q   = thermal_relaxation_error(T1, T2, GATE_2Q)
    therm_2q_2 = therm_2q.expand(2)
    for ctrl in range(n_qubits):
        tgt = (ctrl + 1) % n_qubits
        noise_model.add_quantum_error(therm_2q_2, ["cx"], [ctrl, tgt])

    return noise_model


def get_ibm_device(n_qubits, shots=1024, use_real_qpu=False):
    """
    Returns (PennyLane device, label string).
    Tries AerSimulator + custom noise first; falls back to default.qubit.
    """
    if use_real_qpu:
        print("  NOTE: use_real_qpu=True ignored — simulator-only version.")

    try:
        from qiskit_aer import AerSimulator

        noise_model = _build_noise_model(n_qubits)
        backend     = AerSimulator(noise_model=noise_model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dev = qml.device(
                "qiskit.aer",
                wires=n_qubits,
                backend=backend,
                shots=shots,
            )

        label = (
            f"AerSimulator+DepolarNoise+Thermal "
            f"(1q={DEPOL_1Q:.0e}, 2q={DEPOL_2Q:.0e}, shots={shots}) "
            f"+ DataNoise(σ={ENCODING_NOISE_STD}) "
            f"+ PauliNoise(1q={PAULI_ERROR_PROB_1Q}, 2q={PAULI_ERROR_PROB_2Q})"
        )
        print(f"  Device : {label}")
        return dev, label

    except Exception as e:
        print(f"  WARNING: AerSimulator failed ({e})")
        print(f"  Falling back to default.qubit — "
              f"Layer 1 (data noise) + Layer 2 (Pauli noise) still active.")
        dev   = qml.device("default.qubit", wires=n_qubits, shots=shots)
        label = (
            f"default.qubit+shots={shots} "
            f"+ DataNoise(σ={ENCODING_NOISE_STD}) "
            f"+ PauliNoise(1q={PAULI_ERROR_PROB_1Q}, 2q={PAULI_ERROR_PROB_2Q})"
        )
        return dev, label


# ================================================================
#  ARCH C GENERATOR
#  6 qubits | ring CNOT | RX encoding | RY trainable | 2 layers
#
#  NOISE INSIDE THE CIRCUIT (Layer 2):
#    After every RX  → qml.PauliError with PAULI_ERROR_PROB_1Q
#    After every CNOT → qml.PauliError with PAULI_ERROR_PROB_2Q
#    After every RY  → qml.PauliError with PAULI_ERROR_PROB_1Q
#
#  qml.PauliError applies a random Pauli (X, Y, or Z) with the
#  given probability — this is the circuit-level equivalent of the
#  "inject random values" pattern from the example.
# ================================================================
class GeneratorArchC(torch.nn.Module):

    def __init__(self, n_qubits=6, n_layers=N_LAYERS,
                 shots=1024, use_real_qpu=False):
        super().__init__()
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.training_ = True   # separate flag to avoid nn.Module clash

        self.dev, self.device_label = get_ibm_device(
            n_qubits, shots, use_real_qpu
        )

        # [n_layers, n_qubits] — one RY per qubit per layer
        self.weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )

        n_q           = n_qubits
        p1            = PAULI_ERROR_PROB_1Q
        p2            = PAULI_ERROR_PROB_2Q

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            # --------------------------------------------------
            # Step 1 — RX encoding with Layer 2 Pauli noise
            # --------------------------------------------------
            for w in range(n_q):
                angle = inputs[w] if w < len(inputs) else torch.tensor(0.0)
                qml.RX(angle, wires=w)
                # Layer 2: Pauli error after each RX
                qml.PauliError("X", p1, wires=w)
                qml.PauliError("Z", p1, wires=w)

            # --------------------------------------------------
            # Step 2 — Layers: ring CNOT then RY + Pauli noise
            # --------------------------------------------------
            for l in range(n_layers):
                # Ring CNOT with 2-qubit Pauli noise
                for w in range(n_q):
                    tgt = (w + 1) % n_q
                    qml.CNOT(wires=[w, tgt])
                    # Layer 2: depolarising-style Pauli on both qubits
                    qml.PauliError("X", p2, wires=w)
                    qml.PauliError("Z", p2, wires=tgt)

                # RY trainable with Pauli noise
                for w in range(n_q):
                    qml.RY(weights[l, w], wires=w)
                    qml.PauliError("Y", p1, wires=w)

            return [qml.expval(qml.PauliZ(w)) for w in range(n_q)]

        self.circuit = circuit

    # ------------------------------------------------------------------
    #  Override train/eval so Layer 1 noise tracks nn.Module state
    # ------------------------------------------------------------------
    def train(self, mode=True):
        self.training_ = mode
        return super().train(mode)

    def eval(self):
        self.training_ = False
        return super().eval()

    def forward(self, x):
        x = x.float()

        def _run(xi):
            # Pad input from n_features to n_qubits
            if xi.shape[0] < self.n_qubits:
                pad = torch.zeros(self.n_qubits - xi.shape[0])
                xi  = torch.cat([xi, pad])

            # ---- Layer 1: inject encoding noise ----
            xi = inject_encoding_noise(xi, training=self.training_)

            out = self.circuit(xi, self.weights)
            return torch.stack(out) if isinstance(out, (list, tuple)) else out

        if x.dim() == 1:
            return _run(x)
        return torch.stack([_run(xi) for xi in x])


# ================================================================
#  CLASSICAL DISCRIMINATOR
#  input_dim=6 — generator outputs 6 qubit measurements
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
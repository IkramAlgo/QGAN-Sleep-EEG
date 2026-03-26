# qgan/models_ibm.py
# QGAN Arch C — 6 qubits, ring CNOT, RX->CNOT->RY
#
# NOISE STRATEGY (2 guaranteed layers + 1 optional):
#
#   Layer 1 — DATA-LEVEL NOISE (always active, no backend dependency)
#             Gaussian jitter on RX encoding angles during training.
#             Zero-valued angles get Uniform(0, π/4) replacement.
#             Simulates EEG sensor noise + ADC quantisation error.
#
#   Layer 2 — CIRCUIT-LEVEL BIT-FLIP NOISE (always active)
#             After each gate, with probability p, a random Pauli
#             (X, Y, or Z) is applied via standard RZ/RX/RY rotations
#             (π-rotation = Pauli gate). Works on every PennyLane
#             backend including default.qubit — no channel ops needed.
#
#   Layer 3 — AERSIMULATOR HARDWARE NOISE (optional, best-effort)
#             Depolarizing + thermal relaxation noise model.
#             Used when qiskit-aer is available; graceful fallback
#             to default.qubit (Layers 1+2 remain active either way).

import torch
import warnings
import numpy as np
import pennylane as qml
from qgan.config import N_LAYERS, WEIGHT_INIT_STD


# ================================================================
#  NOISE PARAMETERS
# ================================================================

# Layer 1 — data-level encoding noise
ENCODING_NOISE_STD = 0.15        # Gaussian σ on RX angles (radians)
ZERO_ANGLE_NOISE   = np.pi / 4   # uniform upper bound for zero angles
ZERO_THRESH        = 1e-3        # threshold to treat angle as "zero"

# Layer 2 — circuit-level gate noise probability
PAULI_NOISE_1Q = 0.01            # prob of random Pauli after 1-qubit gate
PAULI_NOISE_2Q = 0.03            # prob of random Pauli after CNOT (per qubit)

# Layer 3 — AerSimulator hardware noise
DEPOL_1Q = 1e-3
DEPOL_2Q = 5e-3
T1       = 50e-6
T2       = 70e-6
GATE_1Q  = 50e-9
GATE_2Q  = 300e-9


# ================================================================
#  LAYER 1 — DATA-LEVEL NOISE
# ================================================================
def inject_encoding_noise(angles: torch.Tensor, training: bool) -> torch.Tensor:
    """
    Corrupt RX encoding angles to simulate EEG sensor + ADC noise.

    - |angle| < ZERO_THRESH  -> replace with Uniform(0, pi/4)
      (mirrors: if images[-1][i] == 0: images[-1][i] = uniform(0, pi/4))
    - otherwise               -> add Gaussian(0, ENCODING_NOISE_STD)

    Only active during training; eval is deterministic.
    """
    if not training:
        return angles

    noisy = angles.clone()
    flat  = noisy.view(-1)
    for i in range(flat.shape[0]):
        val = flat[i].item()
        if abs(val) < ZERO_THRESH:
            flat[i] = float(np.random.uniform(0, ZERO_ANGLE_NOISE))
        else:
            flat[i] = val + float(np.random.normal(0, ENCODING_NOISE_STD))
    return noisy.view(angles.shape)


# ================================================================
#  LAYER 2 — CIRCUIT-LEVEL GATE NOISE
#  Uses only standard rotation gates (RZ, RX, RY) so it works on
#  every PennyLane backend — no channel/mixed-state simulator needed.
#
#  A pi-rotation is equivalent to the corresponding Pauli gate:
#    RZ(pi) = iZ,  RX(pi) = iX,  RY(pi) = iY  (global phase ignored)
#
#  We pick one of {X, Y, Z} uniformly at random and apply it
#  with probability p during training only.
# ================================================================
def apply_pauli_noise(wire: int, prob: float, training: bool):
    """
    With probability `prob`, apply a random Pauli {X, Y, Z} to `wire`
    using pi-rotation gates. No-op during eval.
    """
    if not training:
        return
    if np.random.random() < prob:
        choice = np.random.randint(0, 3)
        if choice == 0:
            qml.RX(np.pi, wires=wire)   # equivalent to Pauli X
        elif choice == 1:
            qml.RY(np.pi, wires=wire)   # equivalent to Pauli Y
        else:
            qml.RZ(np.pi, wires=wire)   # equivalent to Pauli Z


# ================================================================
#  LAYER 3 — AERSIMULATOR NOISE MODEL
# ================================================================
def _build_noise_model(n_qubits):
    """
    Depolarizing + thermal relaxation noise model.
    Uses tensor() to combine two 1-qubit thermal errors into a
    2-qubit error for cx — compatible with older qiskit-aer versions
    that do not support .expand().
    """
    from qiskit_aer.noise import (
        NoiseModel, depolarizing_error, thermal_relaxation_error
    )
    noise_model = NoiseModel()

    # Single-qubit gate noise
    for qubit in range(n_qubits):
        dep_1q      = depolarizing_error(DEPOL_1Q, 1)
        therm_1q    = thermal_relaxation_error(T1, T2, GATE_1Q)
        combined_1q = dep_1q.compose(therm_1q)
        noise_model.add_quantum_error(
            combined_1q,
            ["rx", "ry", "rz", "h", "u1", "u2", "u3"],
            [qubit]
        )

    # Two-qubit depolarizing on each ring pair
    dep_2q = depolarizing_error(DEPOL_2Q, 2)
    for ctrl in range(n_qubits):
        tgt = (ctrl + 1) % n_qubits
        noise_model.add_quantum_error(dep_2q, ["cx"], [ctrl, tgt])

    # Thermal relaxation on cx:
    # tensor() two independent 1-qubit errors -> valid 2-qubit error
    # Avoids .expand() incompatibility on older qiskit-aer versions.
    therm_2q           = thermal_relaxation_error(T1, T2, GATE_2Q)
    therm_2q_tensored  = therm_2q.tensor(therm_2q)   # 2-qubit error
    for ctrl in range(n_qubits):
        tgt = (ctrl + 1) % n_qubits
        noise_model.add_quantum_error(therm_2q_tensored, ["cx"], [ctrl, tgt])

    return noise_model


def get_ibm_device(n_qubits, shots=1024, use_real_qpu=False):
    """Returns (PennyLane device, label string)."""
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
            f"+ DataNoise(sigma={ENCODING_NOISE_STD}) "
            f"+ CircuitNoise(p1q={PAULI_NOISE_1Q}, p2q={PAULI_NOISE_2Q})"
        )
        print(f"  Device : {label}")
        return dev, label

    except Exception as e:
        print(f"  WARNING: AerSimulator failed ({e})")
        print(f"  Falling back to default.qubit — "
              f"Layer 1 (data noise) + Layer 2 (circuit noise) still active.")
        dev   = qml.device("default.qubit", wires=n_qubits, shots=shots)
        label = (
            f"default.qubit+shots={shots} "
            f"+ DataNoise(sigma={ENCODING_NOISE_STD}) "
            f"+ CircuitNoise(p1q={PAULI_NOISE_1Q}, p2q={PAULI_NOISE_2Q})"
        )
        return dev, label


# ================================================================
#  ARCH C GENERATOR
#  6 qubits | ring CNOT | RX encoding | RY trainable | 2 layers
# ================================================================
class GeneratorArchC(torch.nn.Module):

    def __init__(self, n_qubits=6, n_layers=N_LAYERS,
                 shots=1024, use_real_qpu=False):
        super().__init__()
        self.n_qubits    = n_qubits
        self.n_layers    = n_layers
        self.is_training = True   # separate flag; avoids nn.Module.training clash

        self.dev, self.device_label = get_ibm_device(
            n_qubits, shots, use_real_qpu
        )

        self.weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )

        n_q   = n_qubits
        n_lay = n_layers

        # NOTE: apply_pauli_noise uses np.random at Python trace time,
        # so each forward pass gets a fresh independent noise realisation.

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights, training):
            # ---- Step 1: RX encoding + Layer 2 noise ----
            for w in range(n_q):
                angle = inputs[w] if w < len(inputs) else torch.tensor(0.0)
                qml.RX(angle, wires=w)
                apply_pauli_noise(w, PAULI_NOISE_1Q, bool(training))

            # ---- Step 2: Layers ----
            for l in range(n_lay):
                # Ring CNOT + noise on both qubits
                for w in range(n_q):
                    tgt = (w + 1) % n_q
                    qml.CNOT(wires=[w, tgt])
                    apply_pauli_noise(w,   PAULI_NOISE_2Q, bool(training))
                    apply_pauli_noise(tgt, PAULI_NOISE_2Q, bool(training))

                # RY trainable + noise
                for w in range(n_q):
                    qml.RY(weights[l, w], wires=w)
                    apply_pauli_noise(w, PAULI_NOISE_1Q, bool(training))

            return [qml.expval(qml.PauliZ(w)) for w in range(n_q)]

        self.circuit = circuit

    def train(self, mode=True):
        self.is_training = mode
        return super().train(mode)

    def eval(self):
        self.is_training = False
        return super().eval()

    def forward(self, x):
        x = x.float()

        def _run(xi):
            # Pad input to n_qubits
            if xi.shape[0] < self.n_qubits:
                pad = torch.zeros(self.n_qubits - xi.shape[0])
                xi  = torch.cat([xi, pad])

            # Layer 1: data-level encoding noise
            xi = inject_encoding_noise(xi, training=self.is_training)

            # Layers 2+3: circuit with in-circuit Pauli noise
            out = self.circuit(xi, self.weights, self.is_training)
            return torch.stack(out) if isinstance(out, (list, tuple)) else out

        if x.dim() == 1:
            return _run(x)
        return torch.stack([_run(xi) for xi in x])


# ================================================================
#  CLASSICAL DISCRIMINATOR
#  input_dim=6  |  No Sigmoid — WGAN-GP raw scores
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
        )

    def forward(self, x):
        return self.net(x.float())
# qgan/models_ibm.py
# QGAN Arch C — 6 qubits, ring CNOT, RX->CNOT->RY
#
# CLEAN VERSION:
#   ✔ ONLY AerSimulator hardware noise (depolarizing + thermal)
#   ✘ No data-level noise
#   ✘ No circuit-level Pauli noise

import torch
import warnings
import pennylane as qml
from qgan.config import N_LAYERS, WEIGHT_INIT_STD


# ================================================================
#  AER HARDWARE NOISE PARAMETERS
# ================================================================
DEPOL_1Q = 1e-3
DEPOL_2Q = 5e-3
T1       = 50e-6
T2       = 70e-6
GATE_1Q  = 50e-9
GATE_2Q  = 300e-9


# ================================================================
#  AER NOISE MODEL (FIXED — NO DUPLICATION)
# ================================================================
def _build_noise_model(n_qubits):
    from qiskit_aer.noise import (
        NoiseModel, depolarizing_error, thermal_relaxation_error
    )

    noise_model = NoiseModel()

    # ---- 1-QUBIT NOISE ----
    for qubit in range(n_qubits):
        dep_1q   = depolarizing_error(DEPOL_1Q, 1)
        therm_1q = thermal_relaxation_error(T1, T2, GATE_1Q)

        combined_1q = dep_1q.compose(therm_1q)

        noise_model.add_quantum_error(
            combined_1q,
            ["rx", "ry", "rz"],
            [qubit]
        )

    # ---- 2-QUBIT NOISE ----
    dep_2q = depolarizing_error(DEPOL_2Q, 2)

    therm_2q = thermal_relaxation_error(T1, T2, GATE_2Q)
    therm_2q_tensored = therm_2q.tensor(therm_2q)

    combined_2q = dep_2q.compose(therm_2q_tensored)

    for ctrl in range(n_qubits):
        tgt = (ctrl + 1) % n_qubits
        noise_model.add_quantum_error(combined_2q, ["cx"], [ctrl, tgt])

    return noise_model


# ================================================================
#  DEVICE SETUP (AER ONLY)
# ================================================================
def get_ibm_device(n_qubits, shots=1024, use_real_qpu=False):
    if use_real_qpu:
        print("  NOTE: Real QPU disabled — using Aer simulator.")

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
            f"AerSimulator (Hardware Noise Only) | "
            f"Depol1q={DEPOL_1Q:.0e}, Depol2q={DEPOL_2Q:.0e}, shots={shots}"
        )

        print(f"  Device : {label}")
        return dev, label

    except Exception as e:
        print(f"  ERROR: AerSimulator failed: {e}")
        raise RuntimeError("Aer simulator is required for this configuration")


# ================================================================
#  ARCH C GENERATOR (NO ARTIFICIAL NOISE)
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

        self.weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )

        n_q   = n_qubits
        n_lay = n_layers

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):

            # RX encoding
            for w in range(n_q):
                angle = inputs[w] if w < len(inputs) else 0.0
                qml.RX(angle, wires=w)

            # Layers
            for l in range(n_lay):

                # Ring CNOT
                for w in range(n_q):
                    qml.CNOT(wires=[w, (w + 1) % n_q])

                # Trainable RY
                for w in range(n_q):
                    qml.RY(weights[l, w], wires=w)

            return [qml.expval(qml.PauliZ(w)) for w in range(n_q)]

        self.circuit = circuit

    def forward(self, x):
        x = x.float()

        def _run(xi):
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
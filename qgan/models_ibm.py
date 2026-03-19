# qgan/models_ibm.py
# QGAN Arch C — 6 qubits, ring CNOT, RX->CNOT->RY
# Matches CPU Arch C exactly. Uses parameter-shift for QPU compatibility.
# Credentials read from ibm_credentials.txt (written by ibm_quantum_setup.py)

import os
import torch
import warnings
import pennylane as qml
from qgan.config import N_LAYERS, WEIGHT_INIT_STD


def _run_circuit(circuit, x, weights):
    out = circuit(x, weights)
    return torch.stack(out) if isinstance(out, (list, tuple)) else out


def _read_credentials():
    """Read IBM credentials from ibm_credentials.txt"""
    creds = {}
    try:
        with open("ibm_credentials.txt", "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    creds[k.strip()] = v.strip()
    except FileNotFoundError:
        print("  WARNING: ibm_credentials.txt not found.")
        print("  Run ibm_quantum_setup.py first.")
    return creds


def get_ibm_device(n_qubits, shots=1024, use_real_qpu=False):
    """
    Get quantum device.
    use_real_qpu=False  -> Fake127QPulseV1 noise simulation (local test)
    use_real_qpu=True   -> Real IBM QPU via ibm_cloud credentials
    """

    if use_real_qpu:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            creds = _read_credentials()
            api_key = creds.get("api_key") or os.environ.get("IBM_QUANTUM_TOKEN")
            crn     = creds.get("crn")     or os.environ.get("IBM_QUANTUM_CRN")
            backend_name = creds.get("backend", "ibm_torino")

            if not api_key:
                print("  WARNING: No IBM API key found — falling back to noise sim")
                return get_ibm_device(n_qubits, shots, use_real_qpu=False)

            service = QiskitRuntimeService(
                channel="ibm_cloud",
                token=api_key,
                instance=crn
            )

            backend = service.backend(backend_name)
            print(f"  QPU: {backend.name}")

            try:
                pending = backend.status().pending_jobs
                print(f"  Queue: {pending} pending jobs")
            except Exception:
                pass

            dev   = qml.device("qiskit.remote", wires=n_qubits,
                               backend=backend, shots=shots)
            label = f"RealQPU ({backend.name}, {shots} shots)"
            return dev, label

        except Exception as e:
            print(f"  WARNING: Real QPU failed ({e})")
            print(f"  Falling back to noise simulation")
            return get_ibm_device(n_qubits, shots, use_real_qpu=False)

    else:
        # Fake127QPulseV1 noise simulation — works immediately, no credentials
        try:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel

            # Fake127QPulseV1 is 127 qubits — too large for local RAM (MemoryError)
            # FakeNairobi = 7 qubits, same IBM noise characteristics, fits in memory
            # On ARC (real QPU) this branch is never reached anyway
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    from qiskit.providers.fake_provider import FakeNairobi
                    fake_backend = FakeNairobi()
                    label_name   = "FakeNairobi 7-qubit"
                except ImportError:
                    from qiskit.providers.fake_provider import FakeMontreal
                    fake_backend = FakeMontreal()
                    label_name   = "FakeMontreal 27-qubit"
                noise_model = NoiseModel.from_backend(fake_backend)

            dev   = qml.device("qiskit.aer", wires=n_qubits,
                               noise_model=noise_model, shots=shots)
            label = f"NoiseSim {label_name} ({shots} shots)"
            print(f"  Device: {label}")
            return dev, label

        except Exception as e:
            print(f"  WARNING: Noise sim failed ({e}) — using noiseless CPU")
            dev   = qml.device("default.qubit", wires=n_qubits)
            label = "CPU-Sim (noiseless fallback)"
            return dev, label


# ================================================================
#  ARCH C GENERATOR
#  6 qubits | ring CNOT | RX encoding | RY trainable | 2 layers
#  Identical to CPU Arch C — only diff is parameter-shift gradient
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
            # Step 1: RX encoding
            # First 4 inputs = EEG features, qubits 4 and 5 get zero
            for w in range(n_qubits):
                angle = inputs[w] if w < len(inputs) else torch.tensor(0.0)
                qml.RX(angle, wires=w)

            # Step 2: For each layer — ring CNOT then RY trainable
            for l in range(n_layers):
                # Ring CNOT: 0->1->2->3->4->5->0
                for w in range(n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % n_qubits])
                # RY trainable weights
                for w in range(n_qubits):
                    qml.RY(weights[l, w], wires=w)

            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x):
        x = x.float()

        def _run(xi):
            # Pad from 4 features to 6 qubits
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
            # NO Sigmoid
        )

    def forward(self, x):
        return self.net(x.float())
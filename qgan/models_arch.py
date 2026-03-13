# qgan/models_arch.py
# Three new quantum generator architectures for architecture ablation study.
# All use the ClassicalDiscriminator from models.py (hybrid approach).
# Nothing in this file touches the original models.py or any results JSON.
#
# ARCHITECTURE B — Different gate sequence: RZ → RY → CNOT → RX
#   Current (Arch A): RX encode → CNOT → RY weights
#   New    (Arch B): RZ encode → CNOT → RY weights → RX weights
#   Motivation: RZ phase encoding before entanglement vs RX amplitude encoding
#
# ARCHITECTURE C — 6 qubits instead of 4
#   Pads 4-feature input to 6 qubits with two learned bias parameters.
#   More entanglement capacity but higher barren plateau risk.
#   Only run at 4 features (padding used for extra 2 qubits).
#
# ARCHITECTURE D — All-to-all entanglement (full connectivity)
#   Current (Arch A): ring CNOT: 0→1, 1→2, 2→3, 3→0
#   New    (Arch D): all pairs: 0→1, 0→2, 0→3, 1→2, 1→3, 2→3
#   Motivation: captures all pairwise quantum correlations between features
#   Risk: LOW — just more CNOT gates, same qubit count, ~15% slower

import torch
import pennylane as qml

WEIGHT_INIT_STD = 0.01
N_LAYERS = 2


def _run(circuit, x, weights):
    out = circuit(x, weights)
    result = torch.stack(out) if isinstance(out, (list, tuple)) else out
    return result.float()   # cast float64 → float32 for classical discriminator


# =============================================================================
# ARCHITECTURE B — Gate sequence: RZ → RY → CNOT → RX
# =============================================================================

class GeneratorArchB(torch.nn.Module):
    """
    Arch B generator: RZ encoding → CNOT entangle → RY weights → RX weights
    Different from Arch A (RX encode → CNOT → RY weights).
    RZ encodes phase information vs RX encodes amplitude.
    Two rotation gates per layer (RY then RX) gives more expressibility.
    """

    def __init__(self, n_qubits=4, n_layers=N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device   = qml.device("default.qubit", wires=n_qubits)
        # two weight matrices per layer: RY weights and RX weights
        self.ry_weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )
        self.rx_weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(inputs, ry_w, rx_w):
            # --- Encoding: RZ (phase encoding, different from Arch A RX) ---
            for w in range(n_qubits):
                qml.RZ(inputs[w], wires=w)
            # --- Trainable layers: CNOT then RY then RX ---
            for l in range(n_layers):
                # ring entanglement (same as Arch A for fair gate-sequence comparison)
                for w in range(n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % n_qubits])
                # RY rotation
                for w in range(n_qubits):
                    qml.RY(ry_w[l, w], wires=w)
                # RX rotation (extra rotation — more expressive)
                for w in range(n_qubits):
                    qml.RX(rx_w[l, w], wires=w)
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x):
        if x.dim() == 1:
            return _run(self.circuit, x, self.ry_weights, self.rx_weights)
        return torch.stack([
            _run(self.circuit, xi, self.ry_weights, self.rx_weights) for xi in x
        ])

    def parameters(self):
        return iter([self.ry_weights, self.rx_weights])


# Fix: override _run for 3-arg circuits
def _run3(circuit, x, w1, w2):
    out = circuit(x, w1, w2)
    result = torch.stack(out) if isinstance(out, (list, tuple)) else out
    return result.float()   # cast float64 → float32 for classical discriminator


class GeneratorArchB(torch.nn.Module):
    """Arch B: RZ encoding + dual rotation per layer (RY+RX)."""

    def __init__(self, n_qubits=4, n_layers=N_LAYERS):
        super().__init__()
        self.n_qubits   = n_qubits
        self.n_layers   = n_layers
        self.device     = qml.device("default.qubit", wires=n_qubits)
        self.ry_weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )
        self.rx_weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(inputs, ry_w, rx_w):
            for w in range(n_qubits):
                qml.RZ(inputs[w], wires=w)          # phase encoding
            for l in range(n_layers):
                for w in range(n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % n_qubits])
                for w in range(n_qubits):
                    qml.RY(ry_w[l, w], wires=w)
                for w in range(n_qubits):
                    qml.RX(rx_w[l, w], wires=w)     # extra rotation gate
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x):
        if x.dim() == 1:
            return _run3(self.circuit, x, self.ry_weights, self.rx_weights)
        return torch.stack([
            _run3(self.circuit, xi, self.ry_weights, self.rx_weights) for xi in x
        ])

    def named_parameters(self, prefix='', recurse=True):
        yield 'ry_weights', self.ry_weights
        yield 'rx_weights', self.rx_weights

    def parameters(self, recurse=True):
        return iter([self.ry_weights, self.rx_weights])


# =============================================================================
# ARCHITECTURE C — 6 qubits (4 feature qubits + 2 ancilla qubits)
# =============================================================================

class GeneratorArchC(torch.nn.Module):
    """
    Arch C: 6-qubit generator for 4-feature input.
    Input features go into qubits 0-3 via RX encoding.
    Qubits 4-5 are ancilla — initialized with learned bias parameters.
    All 6 qubits participate in ring entanglement.
    More qubits = richer Hilbert space = potentially better variance matching.
    Risk: barren plateau gets worse with more qubits.
    Output: only qubits 0-3 measured (to match 4-feature output).
    """

    N_QUBITS_TOTAL = 6   # total qubits in circuit
    N_FEATURES     = 4   # features in + features out

    def __init__(self, n_features=4, n_layers=N_LAYERS):
        super().__init__()
        self.n_features    = n_features
        self.n_qubits      = self.N_QUBITS_TOTAL
        self.n_layers      = n_layers
        self.device        = qml.device("default.qubit", wires=self.n_qubits)
        # weights for all 6 qubits
        self.weights       = torch.nn.Parameter(
            torch.randn(n_layers, self.n_qubits) * WEIGHT_INIT_STD
        )
        # learned initialization angles for the 2 ancilla qubits
        self.ancilla_bias  = torch.nn.Parameter(
            torch.zeros(2)
        )

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights, ancilla):
            # encode 4 input features into qubits 0-3
            for w in range(n_features):
                qml.RX(inputs[w], wires=w)
            # ancilla qubits 4 and 5 — learned initialization
            qml.RY(ancilla[0], wires=4)
            qml.RY(ancilla[1], wires=5)
            # trainable layers — all 6 qubits entangled
            for l in range(n_layers):
                for w in range(self.n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % self.n_qubits])
                for w in range(self.n_qubits):
                    qml.RY(weights[l, w], wires=w)
            # measure only the 4 feature qubits
            return [qml.expval(qml.PauliZ(w)) for w in range(n_features)]

        self.circuit = circuit

    def _fwd(self, x):
        out = self.circuit(x, self.weights, self.ancilla_bias)
        result = torch.stack(out) if isinstance(out, (list, tuple)) else out
        return result.float()   # cast float64 → float32

    def forward(self, x):
        if x.dim() == 1:
            return self._fwd(x)
        return torch.stack([self._fwd(xi) for xi in x])

    def parameters(self, recurse=True):
        return iter([self.weights, self.ancilla_bias])


# =============================================================================
# ARCHITECTURE D — All-to-all entanglement (full connectivity)
# =============================================================================

class GeneratorArchD(torch.nn.Module):
    """
    Arch D: all-to-all CNOT entanglement.
    Instead of ring (0→1, 1→2, 2→3, 3→0), uses all pairs:
      0→1, 0→2, 0→3, 1→2, 1→3, 2→3
    For n=4 qubits: 6 CNOT gates per layer vs 4 CNOT gates in Arch A.
    Captures ALL pairwise correlations between EEG features.
    Risk: LOW. Same weights, same qubit count. ~15% slower due to extra CNOTs.
    """

    def __init__(self, n_qubits=4, n_layers=N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device   = qml.device("default.qubit", wires=n_qubits)
        self.weights  = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )
        # precompute all-to-all CNOT pairs: (control, target) for control < target
        self.cnot_pairs = [
            (i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)
        ]

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # same encoding as Arch A (RX) — isolates effect of entanglement change
            for w in range(n_qubits):
                qml.RX(inputs[w], wires=w)
            for l in range(n_layers):
                # ALL-TO-ALL CNOT: every qubit entangled with every other qubit
                for (ctrl, tgt) in self.cnot_pairs:
                    qml.CNOT(wires=[ctrl, tgt])
                # same RY trainable weights as Arch A
                for w in range(n_qubits):
                    qml.RY(weights[l, w], wires=w)
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x):
        if x.dim() == 1:
            return _run(self.circuit, x, self.weights)
        return torch.stack([
            _run(self.circuit, xi, self.weights) for xi in x
        ])

    def parameters(self, recurse=True):
        return iter([self.weights])
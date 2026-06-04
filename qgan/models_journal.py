# qgan/models_journal.py
# Journal Extension — Multi-Baseline Quantum Generator Study
#
# ── WHAT CHANGED vs PREVIOUS VERSION (and WHY) ─────────────────────────────
#
#  FIX 1 (Reviewer 2, Weakness 1 — weak classical baseline):
#    ClassicalBCEGenerator is NOW its own small MLP matching the ACTUAL
#    conference paper architecture (z→16→n_features, BatchNorm, ReLU).
#    Previously it was identical to ClassicalWGANGenerator — meaning the
#    "conference baseline" and the "new fair baseline" were the same model,
#    which made no sense. Now:
#      - ClassicalBCEGenerator   = conference architecture (small, BCE loss)
#      - ClassicalWGANGenerator  = deeper fair baseline (same loss as QGAN)
#      - DCGANStyleGenerator     = modern convolutional baseline
#    This gives three structurally distinct classical generators.
#
#  FIX 2 (Reviewer 2, Weakness 1 — parameter count not saved):
#    build_models() now RETURNS n_params_gen and n_params_disc so
#    train_journal.py can save them to the results JSON.
#    Nature reviewers require a parameter comparison table.
#
#  FIX 3 (DCGANStyleGenerator dimension bug):
#    conv_out_size is now computed DYNAMICALLY via a dry-run forward pass
#    through the conv stack instead of being hardcoded as base_channels*7.
#    Previously the hardcoded value would silently break if base_channels
#    was changed.
#
#  FIX 4 (ZNE import fragility):
#    _build_zne_circuit() now catches ALL import failures and logs a clear
#    error with PennyLane version info, then falls back gracefully to the
#    base circuit rather than crashing the entire run.
#
#  FIX 5 (Expressibility metric — Nature requirement):
#    compute_expressibility() added. Implements the Sim et al. (2019)
#    expressibility measure: KL divergence between the Haar-random fidelity
#    distribution and the circuit's sampled fidelity distribution.
#    Higher expressibility = quantum circuit covers more of Hilbert space.
#    Run ONCE per architecture; results saved to expressibility.json.
#
# ── CONDITION OVERVIEW ──────────────────────────────────────────────────────
#   Condition 1: default.qubit, noiseless, backprop
#   Condition 2: default.qubit + Gaussian input noise, backprop
#   Condition 3: FakeNairobi AerSimulator, SPSA
#   Condition 4: FakeNairobi AerSimulator + ZNE, SPSA
#
# ── ARCHITECTURE ────────────────────────────────────────────────────────────
#   Arch C: n_features + 2 ancilla qubits, ring CNOT, RX encoding, RY trainable
#   statistical (4 feat)  →  6 qubits  (4+2 ancilla)
#   spectral   (5 feat)  →  7 qubits  (5+2 ancilla)
#   combined   (9 feat)  → 11 qubits  (9+2 ancilla)

import os
import warnings
import torch
import torch.nn as nn
import pennylane as qml
from qgan.config import N_LAYERS, WEIGHT_INIT_STD

NUM_THREADS = int(os.getenv("OMP_NUM_THREADS", "38"))
torch.set_num_threads(NUM_THREADS)

QPU_SHOTS         = 128
SPSA_H            = 0.05
ZNE_SCALE_FACTORS = [1, 3, 5]

# ── Condition keys ─────────────────────────────────────────────────────────
CONDITION_SIMULATOR    = "simulator"
CONDITION_DATA_NOISE   = "simulator_datanoise"
CONDITION_QPU_SIM      = "qpu_noiseless"
CONDITION_QPU_ZNE      = "qpu_noise_zne"

ALL_CONDITIONS = [
    CONDITION_SIMULATOR,
    CONDITION_DATA_NOISE,
    CONDITION_QPU_SIM,
    CONDITION_QPU_ZNE,
]

CONDITION_LABELS = {
    CONDITION_SIMULATOR:  "Simulator (noiseless)",
    CONDITION_DATA_NOISE: "Simulator + DataNoise",
    CONDITION_QPU_SIM:    "QPU-Sim (FakeNairobi)",
    CONDITION_QPU_ZNE:    "QPU-Sim + ZNE",
}

CONDITION_GRAD_METHOD = {
    CONDITION_SIMULATOR:  "backprop",
    CONDITION_DATA_NOISE: "backprop",
    CONDITION_QPU_SIM:    "spsa",
    CONDITION_QPU_ZNE:    "spsa",
}

# ── Generator type keys ────────────────────────────────────────────────────
GEN_CLASSICAL_BCE  = "classical_bce"    # conference baseline (small MLP, BCE)
GEN_CLASSICAL_WGAN = "classical_wgan"   # fair baseline — same loss as QGAN
GEN_DCGAN          = "dcgan"            # modern convolutional baseline
GEN_QUANTUM        = "quantum"          # quantum circuit generator

ALL_GENERATOR_TYPES = [
    GEN_CLASSICAL_BCE,
    GEN_CLASSICAL_WGAN,
    GEN_DCGAN,
    GEN_QUANTUM,
]

GENERATOR_LABELS = {
    GEN_CLASSICAL_BCE:  "Classical GAN (BCE, conference)",
    GEN_CLASSICAL_WGAN: "Classical WGAN-GP (fair)",
    GEN_DCGAN:          "DCGAN-style WGAN-GP",
    GEN_QUANTUM:        "QWGAN-GP",
}

# ================================================================
#  DEVICE CACHE
# ================================================================
_device_cache: dict = {}


def _get_device(condition: str, n_qubits: int = 6):
    """
    Return (pennylane_device, diff_method) for the given condition.
    Cached after first construction.
    Falls back gracefully to default.qubit if Qiskit/FakeNairobi unavailable.
    """
    cache_key = (condition, n_qubits)
    if cache_key in _device_cache:
        return _device_cache[cache_key]

    if condition in (CONDITION_SIMULATOR, CONDITION_DATA_NOISE):
        dev    = qml.device("default.qubit", wires=n_qubits)
        result = (dev, "backprop")
    else:
        try:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel
            # Try both old and new Qiskit fake provider APIs (FIX 4 style)
            FakeNairobi = None
            for import_path in [
                "qiskit.providers.fake_provider",
                "qiskit_ibm_runtime.fake_provider",
                "qiskit_aer.primitives",
            ]:
                try:
                    mod = __import__(import_path, fromlist=["FakeNairobi"])
                    FakeNairobi = getattr(mod, "FakeNairobi", None)
                    if FakeNairobi is not None:
                        break
                except ImportError:
                    continue

            if FakeNairobi is None:
                raise ImportError("FakeNairobi not found in any Qiskit package.")

            noise_model = NoiseModel.from_backend(FakeNairobi())
            aer_backend = AerSimulator(
                noise_model=noise_model,
                max_parallel_threads=NUM_THREADS,
                max_parallel_experiments=1,
                statevector_parallel_threshold=14,
            )
            dev    = qml.device("qiskit.aer", wires=n_qubits,
                                backend=aer_backend, shots=QPU_SHOTS)
            result = (dev, "spsa")

        except Exception as exc:
            warnings.warn(
                f"[models_journal] FakeNairobi/AerSimulator unavailable: {exc}\n"
                f"PennyLane version: {qml.__version__}\n"
                f"Falling back to default.qubit (noiseless). "
                f"QPU conditions will NOT reflect hardware noise.",
                stacklevel=2,
            )
            dev    = qml.device("default.qubit", wires=n_qubits)
            result = (dev, "backprop")

    _device_cache[cache_key] = result
    return result


# ================================================================
#  ARCH C QUANTUM CIRCUIT
#  Topology: RX encoding → ring CNOT → RY trainable (per layer)
#  Measurement: PauliZ on all n_qubits
# ================================================================
def _build_circuit(dev, diff_method: str, n_qubits: int, n_layers: int):
    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        for w in range(n_qubits):
            angle = inputs[w] if w < inputs.shape[0] else torch.tensor(0.0)
            qml.RX(angle, wires=w)
        for layer in range(n_layers):
            for w in range(n_qubits):
                qml.CNOT(wires=[w, (w + 1) % n_qubits])
            for w in range(n_qubits):
                qml.RY(weights[layer, w], wires=w)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
    return circuit


def _build_zne_circuit(base_circuit, n_qubits: int, n_layers: int):
    """
    FIX 4: Catches ALL import failures with version info, falls back gracefully.
    """
    try:
        from pennylane.transforms import mitigate_with_zne, richardson_extrapolate
        extrapolator = richardson_extrapolate
    except ImportError:
        try:
            from pennylane.transforms import mitigate_with_zne, poly_extrapolate
            extrapolator = poly_extrapolate
        except ImportError:
            warnings.warn(
                f"ZNE not available in PennyLane {qml.__version__}. "
                f"Requires >= 0.28. Falling back to base circuit (no ZNE). "
                f"ZNE results will be INVALID for this run.",
                stacklevel=2,
            )
            return base_circuit, False

    try:
        zne_circuit = mitigate_with_zne(
            base_circuit,
            scale_factors=ZNE_SCALE_FACTORS,
            extrapolate=extrapolator,
        )
        return zne_circuit, True
    except Exception as e:
        warnings.warn(
            f"ZNE circuit construction failed: {e}. "
            f"Falling back to base circuit.",
            stacklevel=2,
        )
        return base_circuit, False


# ================================================================
#  QUANTUM GENERATOR — GeneratorJournal
# ================================================================
class GeneratorJournal(nn.Module):
    """
    Arch C quantum generator.
    n_qubits = n_features + 2 ancilla (always 2 ancilla).

    statistical (4 feat) → 6 qubits
    spectral    (5 feat) → 7 qubits
    combined    (9 feat) → 11 qubits

    The 2 ancilla qubits extend the Hilbert space without increasing
    output dimensionality (Biamonte et al. 2017).
    Forward: [batch, n_features] latent noise → [batch, n_qubits] PauliZ values
    """

    def __init__(self, condition=CONDITION_SIMULATOR,
                 n_qubits=6, n_layers=N_LAYERS, n_features=4):
        super().__init__()
        self.condition  = condition
        self.n_qubits   = n_qubits
        self.n_layers   = n_layers
        self.n_features = n_features

        dev, diff_method = _get_device(condition, n_qubits)
        self.diff_method = diff_method
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits) * WEIGHT_INIT_STD
        )

        base_circuit = _build_circuit(dev, diff_method, n_qubits, n_layers)
        if condition == CONDITION_QPU_ZNE:
            self._circuit, self._zne_active = _build_zne_circuit(
                base_circuit, n_qubits, n_layers
            )
        else:
            self._circuit    = base_circuit
            self._zne_active = False

    def _run_single(self, xi: torch.Tensor) -> torch.Tensor:
        if xi.shape[0] < self.n_qubits:
            pad = torch.zeros(self.n_qubits - xi.shape[0],
                              dtype=xi.dtype, device=xi.device)
            xi = torch.cat([xi, pad])
        out = self._circuit(xi, self.weights)
        return torch.stack(out) if isinstance(out, (list, tuple)) else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.dim() == 1:
            return self._run_single(x)
        return torch.stack([self._run_single(xi) for xi in x])

    def extra_repr(self):
        zne = f" | ZNE={ZNE_SCALE_FACTORS}" if self._zne_active else ""
        return (f"condition={self.condition}, n_qubits={self.n_qubits}, "
                f"n_layers={self.n_layers}, diff_method={self.diff_method}{zne}")


# ================================================================
#  CLASSICAL BASELINE 1: ClassicalBCEGenerator — FIX 1
#
#  This is the ACTUAL conference paper architecture.
#  Small MLP: z→16→n_features, BatchNorm, ReLU, Tanh output.
#  Trained with BCE loss (not WGAN-GP).
#  Kept SEPARATE from ClassicalWGANGenerator so the two baselines
#  are architecturally distinct and the results are interpretable.
#
#  Why this matters: Previously this used ClassicalWGANGenerator's
#  architecture, which made the "conference BCE baseline" and the
#  "new WGAN-GP fair baseline" structurally identical — a reviewer
#  would immediately notice the results would be nearly identical too.
# ================================================================
class ClassicalBCEGenerator(nn.Module):
    """
    Conference paper architecture: small MLP, BCE loss.
    z(n_features) → Linear(16) → BatchNorm → ReLU → Linear(n_features) → Tanh
    Trained with BCE loss; this is the ORIGINAL baseline from DSN 2026.
    """

    def __init__(self, n_features: int = 4):
        super().__init__()
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, n_features),
            nn.Tanh(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.float())

    def extra_repr(self):
        return f"n_features={self.n_features}, loss=BCE, arch=conference-original"


# ================================================================
#  CLASSICAL BASELINE 2: ClassicalWGANGenerator (Reviewer 2 fix)
#
#  The FAIR comparison to QGAN:
#    - Same WGAN-GP loss (not BCE like the conference paper)
#    - Same discriminator
#    - Deeper MLP: z→32→64→32→n_features
#    - NO BatchNorm (incompatible with WGAN-GP gradient penalty)
#    - LeakyReLU throughout, Tanh output
#
#  This removes the loss-function confound from the conference comparison.
#  If QGAN still outperforms this model, the advantage is architectural.
# ================================================================
class ClassicalWGANGenerator(nn.Module):
    """
    Deeper fair-comparison MLP, WGAN-GP loss.
    z(n_features) → 32 → 64 → 32 → n_features, LeakyReLU, Tanh, no BatchNorm.
    """

    def __init__(self, n_features: int = 4):
        super().__init__()
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, n_features),
            nn.Tanh(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.float())

    def extra_repr(self):
        return f"n_features={self.n_features}, loss=WGAN-GP, arch=deeper-fair"


# ================================================================
#  CLASSICAL BASELINE 3: DCGANStyleGenerator (Reviewer 2 "modern method")
#
#  DCGAN (Radford et al. 2015) for tabular 1D EEG features.
#  FIX 3: conv_out_size computed DYNAMICALLY via a dry-run, not hardcoded.
#  BatchNorm allowed in GENERATOR (only prohibited in WGAN-GP discriminator).
# ================================================================
class DCGANStyleGenerator(nn.Module):
    """
    DCGAN-inspired 1D convolutional generator for tabular EEG features.
    FIX 3: output linear size computed dynamically — safe for any base_channels.
    Reference: Radford et al. (2015) DCGAN.
    """

    def __init__(self, n_features: int = 4, base_channels: int = 32):
        super().__init__()
        self.n_features    = n_features
        self.base_channels = base_channels

        self.project = nn.Sequential(
            nn.Linear(n_features, base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(base_channels * 2, base_channels,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # FIX 3: compute conv output size dynamically via dry-run
        with torch.no_grad():
            dummy = torch.zeros(1, base_channels * 4, 1)
            conv_out = self.conv_blocks(dummy)
            conv_out_size = conv_out.shape[1] * conv_out.shape[2]

        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, n_features),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose1d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        x = self.project(z)
        x = x.unsqueeze(-1)
        x = self.conv_blocks(x)
        x = self.output_layer(x)
        return x

    def extra_repr(self):
        return (f"n_features={self.n_features}, "
                f"base_channels={self.base_channels}, loss=WGAN-GP, style=DCGAN-1D")


# ================================================================
#  CLASSICAL DISCRIMINATOR — shared across ALL models
#  Input:  [batch, input_dim]
#  Output: [batch, 1] — raw Wasserstein score (no Sigmoid)
#  No BatchNorm: required by WGAN-GP gradient penalty
# ================================================================
class ClassicalDiscriminator(nn.Module):
    """
    Wasserstein critic — shared across all model variants.
    32 → 16 → 1, LeakyReLU, Dropout(0.3), no Sigmoid, no BatchNorm.
    """

    def __init__(self, input_dim: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())

    def extra_repr(self):
        return f"input_dim={self.input_dim}, loss=WGAN-GP (no sigmoid)"


# ================================================================
#  MODEL FACTORY — FIX 2: returns parameter counts
# ================================================================
def build_models(condition: str,
                 n_qubits: int = 6,
                 n_features: int = 4,
                 generator_type: str = GEN_QUANTUM) -> tuple:
    """
    Construct (generator, ClassicalDiscriminator, n_params_gen, n_params_disc).

    FIX 2: Returns parameter counts for logging to results JSON.

    Args:
        condition      : CONDITION_* (used only when generator_type=GEN_QUANTUM)
        n_qubits       : qubit count for quantum generator (n_features + 2 ancilla)
        n_features     : EEG feature count
        generator_type : GEN_CLASSICAL_BCE | GEN_CLASSICAL_WGAN |
                         GEN_DCGAN | GEN_QUANTUM

    Returns:
        gen            : generator model
        disc           : ClassicalDiscriminator
        n_params_gen   : int — trainable parameter count of generator
        n_params_disc  : int — trainable parameter count of discriminator

    NOTE on disc input_dim:
        Quantum generator outputs n_qubits values (Pauli-Z per qubit).
        Classical generators output n_features values.
        Discriminator input_dim must match what it will receive.
    """
    if generator_type == GEN_QUANTUM:
        gen = GeneratorJournal(
            condition=condition,
            n_qubits=n_qubits,
            n_layers=N_LAYERS,
            n_features=n_features,
        )
        disc_input_dim = n_qubits   # quantum output is n_qubits wide

    elif generator_type == GEN_CLASSICAL_WGAN:
        gen = ClassicalWGANGenerator(n_features=n_features)
        disc_input_dim = n_features

    elif generator_type == GEN_DCGAN:
        gen = DCGANStyleGenerator(n_features=n_features)
        disc_input_dim = n_features

    elif generator_type == GEN_CLASSICAL_BCE:
        # FIX 1: use the ACTUAL conference architecture (small MLP)
        gen = ClassicalBCEGenerator(n_features=n_features)
        disc_input_dim = n_features

    else:
        raise ValueError(f"Unknown generator_type='{generator_type}'. "
                         f"Choose from: {ALL_GENERATOR_TYPES}")

    disc = ClassicalDiscriminator(input_dim=disc_input_dim)

    # FIX 2: count and return parameters
    n_params_gen  = sum(p.numel() for p in gen.parameters()  if p.requires_grad)
    n_params_disc = sum(p.numel() for p in disc.parameters() if p.requires_grad)

    grad_info = (CONDITION_GRAD_METHOD.get(condition, "N/A")
                 if generator_type == GEN_QUANTUM else "backprop")
    print(f"    [build_models] "
          f"gen={GENERATOR_LABELS.get(generator_type, generator_type)} | "
          f"cond={condition} | grad={grad_info} | "
          f"params: gen={n_params_gen} disc={n_params_disc}")

    return gen, disc, n_params_gen, n_params_disc


# ================================================================
#  FIX 5: EXPRESSIBILITY METRIC (Sim et al. 2019)
#  Measures how well the circuit covers the space of all unitaries
#  (Hilbert space coverage).  Higher = more expressive circuit.
#
#  Method: sample N pairs of random parameter vectors, compute
#  fidelity |<psi1|psi2>|^2, compare KL divergence of this
#  distribution to the Haar-random (uniform) distribution.
#
#  Run ONCE per architecture — saved to expressibility.json.
#  Does NOT need training data.
# ================================================================
def compute_expressibility(n_qubits: int = 6,
                            n_layers: int = None,
                            n_samples: int = 1000,
                            n_bins: int = 75) -> dict:
    """
    Compute expressibility score for Arch C circuit (Sim et al. 2019).

    Args:
        n_qubits  : number of qubits
        n_layers  : number of variational layers (default: N_LAYERS from config)
        n_samples : number of random parameter pairs to sample
        n_bins    : histogram bins for KL divergence

    Returns:
        dict with keys: expressibility_kl, n_qubits, n_layers, n_params,
                        fidelity_mean, fidelity_std
    """
    import numpy as np
    if n_layers is None:
        n_layers = N_LAYERS

    #dev = qml.device("default.qubit", wires=n_qubits)  # BEFORE (slow):
    # AFTER (10-50× faster for noiseless):
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="numpy")
    def statevec(inputs, weights):
        for w in range(n_qubits):
            angle = inputs[w] if w < len(inputs) else 0.0
            qml.RX(angle, wires=w)
        for layer in range(n_layers):
            for w in range(n_qubits):
                qml.CNOT(wires=[w, (w + 1) % n_qubits])
            for w in range(n_qubits):
                qml.RY(weights[layer, w], wires=w)
        return qml.state()

    n_params = n_layers * n_qubits
    fidelities = []

    for _ in range(n_samples):
        inputs1  = np.random.uniform(0, 2 * np.pi, n_qubits)
        weights1 = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits))
        inputs2  = np.random.uniform(0, 2 * np.pi, n_qubits)
        weights2 = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits))

        psi1 = statevec(inputs1, weights1)
        psi2 = statevec(inputs2, weights2)
        fid  = float(np.abs(np.dot(np.conj(psi1), psi2)) ** 2)
        fidelities.append(fid)

    fidelities = np.array(fidelities)

    # KL divergence vs Haar-random distribution
    # Haar fidelity PDF: P_Haar(F) = (2^n - 1)(1-F)^(2^n - 2)
    dim     = 2 ** n_qubits
    bins    = np.linspace(0, 1, n_bins + 1)
    bin_mids = 0.5 * (bins[:-1] + bins[1:])
    bin_w   = bins[1] - bins[0]

    # Circuit histogram
    hist, _ = np.histogram(fidelities, bins=bins, density=True)
    P_circ  = hist * bin_w + 1e-10   # avoid log(0)
    P_circ  = P_circ / P_circ.sum()

    # Haar PDF at bin midpoints
    P_haar = (dim - 1) * ((1 - bin_mids) ** (dim - 2)) * bin_w
    P_haar = P_haar + 1e-10
    P_haar = P_haar / P_haar.sum()

    kl_div = float(np.sum(P_circ * np.log(P_circ / P_haar)))

    result = {
        "expressibility_kl":   round(kl_div, 6),
        "note":                "Lower KL = more expressive (closer to Haar-random)",
        "n_qubits":            n_qubits,
        "n_layers":            n_layers,
        "n_trainable_params":  n_params,
        "hilbert_space_dim":   int(dim),
        "fidelity_mean":       round(float(fidelities.mean()), 6),
        "fidelity_std":        round(float(fidelities.std()),  6),
        "n_samples":           n_samples,
        "reference":           "Sim et al. (2019) Expressibility and entangling "
                               "capability of parameterized quantum circuits for "
                               "hybrid quantum-classical algorithms. Adv. Quantum "
                               "Technol. 2(12):1900070.",
    }

    print(f"  Expressibility (Arch C, {n_qubits} qubits, {n_layers} layers):")
    print(f"    KL divergence: {kl_div:.6f}  "
          f"(lower = more expressive, closer to Haar-random)")
    print(f"    Hilbert space dim: 2^{n_qubits} = {dim}")
    return result


def run_expressibility_sweep(feature_sets: dict = None,
                              output_file: str = "expressibility.json") -> dict:
    """
    Compute expressibility for each feature set (different qubit counts).
    Saves results to expressibility.json for the paper.

    Args:
        feature_sets : dict of {name: n_features} — default covers all three
        output_file  : output JSON path
    """
    import json

    if feature_sets is None:
        feature_sets = {
            "statistical": 4,   # 6 qubits
            "spectral":    5,   # 7 qubits
            "combined":    9,   # 11 qubits
        }

    results = {}
    for fs_name, n_feat in feature_sets.items():
        n_qubits = n_feat + 2   # always 2 ancilla
        print(f"\n  Computing expressibility: {fs_name} "
              f"({n_feat} features → {n_qubits} qubits)")
        results[fs_name] = compute_expressibility(
            n_qubits=n_qubits, n_samples=500
        )

    # Also compute for a parameter-matched classical MLP for comparison
    # Classical MLP with same parameter count as quantum circuit
    print(f"\n  (Expressibility is circuit-specific; classical MLPs have "
          f"infinite expressibility by construction — no KL upper bound.)")
    results["_note"] = (
        "Classical MLP generators are universal function approximators "
        "and do not have a bounded expressibility in the Sim et al. sense. "
        "The quantum KL score measures how uniformly Arch C samples the "
        "Haar-random unitary space. Lower = better coverage."
    )

    with open(output_file, "w") as f:
        import json
        json.dump(results, f, indent=2)
    print(f"\n  Expressibility saved: {output_file}")
    return results


if __name__ == "__main__":
    # Run expressibility sweep standalone
    run_expressibility_sweep()
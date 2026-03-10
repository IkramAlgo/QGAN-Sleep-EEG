# run_setup.py
# Run this FIRST before anything else
# It installs all required packages and checks versions

import subprocess
import sys

packages = [
    "qiskit==1.2.4",
    "qiskit-aer==0.15.1",
    "pennylane-qiskit==0.38.0",
    "pennylane==0.38.0",
    "torch",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "pyedflib",
]

print("Installing required packages...")
for pkg in packages:
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=False)

print("\nVerifying installations:")
imports = [
    ("pennylane",       "qml"),
    ("qiskit",          "qiskit"),
    ("qiskit_aer",      "qiskit_aer"),
    ("pennylane_qiskit","pennylane_qiskit"),
    ("torch",           "torch"),
    ("sklearn",         "sklearn"),
]
for mod, alias in imports:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "ok")
        print(f"  ✓ {mod} {ver}")
    except ImportError:
        print(f"  ✗ {mod} — FAILED, install manually")
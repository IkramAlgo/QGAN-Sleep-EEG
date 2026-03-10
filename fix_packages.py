# fix_packages.py
# Run this to install exact correct versions
# python fix_packages.py

import subprocess, sys

print("Fixing package versions for IBM QPU + noise sim...")

# uninstall all quantum packages first for clean slate
remove = [
    "qiskit", "qiskit-ibm-runtime", "qiskit-aer",
    "pennylane", "pennylane-qiskit", "qiskit-ibm-provider"
]
for pkg in remove:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", pkg, "-y", "-q"],
                   capture_output=True)
    print(f"  Removed: {pkg}")

# install exact compatible stack
# qiskit-ibm-runtime 0.29+ has SamplerV2
# pennylane-qiskit 0.39+ has qiskit.ibm device
install = [
    "sympy==1.12",
    "qiskit==1.3.2",
    "qiskit-ibm-runtime==0.29.0",
    "qiskit-aer==0.15.1",
    "pennylane==0.39.0",
    "pennylane-qiskit==0.39.0",
]
for pkg in install:
    print(f"  Installing {pkg}...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"  WARNING: {r.stderr[-200:]}")
    else:
        print(f"  OK")

print("\nVerifying...")
checks = [
    ("pennylane",          "PennyLane"),
    ("qiskit",             "Qiskit"),
    ("qiskit_ibm_runtime", "Qiskit IBM Runtime"),
    ("qiskit_aer",         "Qiskit Aer"),
]
for mod, name in checks:
    try:
        m = __import__(mod)
        print(f"  OK {name} {getattr(m,'__version__','?')}")
    except ImportError as e:
        print(f"  FAIL {name}: {e}")

print("\nNow run: python test_ibm_device.py")
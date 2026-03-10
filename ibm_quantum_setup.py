# ibm_quantum_setup.py
# Run: python ibm_quantum_setup.py

import sys

IBM_API_KEY = "oWVvlq2AC8W5eenWZnKySERczetRDrbi2aQwdzJpbhBQ"
IBM_CRN     = "crn:v1:bluemix:public:quantum-computing:us-east:a/65a9c8314a3f4121bcd47b86eb75ce3d:ce5e0649-27be-4809-990f-9302ec92448e::"

print("Verifying imports...")
for mod, name in [("pennylane","PennyLane"),("qiskit","Qiskit"),
                  ("qiskit_ibm_runtime","Qiskit IBM Runtime"),("qiskit_aer","Qiskit Aer")]:
    try:
        m = __import__(mod)
        print(f"  OK {name} {getattr(m,'__version__','ok')}")
    except ImportError as e:
        print(f"  FAILED {name}: {e}"); sys.exit(1)

print("\nConnecting to IBM Quantum Platform...")
try:
    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(
        channel="ibm_cloud", token=IBM_API_KEY,
        instance=IBM_CRN, overwrite=True
    )
    service  = QiskitRuntimeService(channel="ibm_cloud",
                                    token=IBM_API_KEY, instance=IBM_CRN)
    backends = service.backends(operational=True, simulator=False)

    print(f"  Connected! {len(backends)} backends found:")
    backend_names = []
    for b in backends:
        name = b.name
        backend_names.append(name)
        try:
            pending = b.status().pending_jobs
        except Exception:
            pending = 0
        print(f"    {name:<35} pending_jobs={pending}")

    # ibm_torino, ibm_fez, ibm_marrakesh are all 133-156 qubit devices
    # pick ibm_torino as default (most modern, heron processor)
    # all have way more than 4 qubits we need
    preferred = ["ibm_torino", "ibm_fez", "ibm_marrakesh"]
    best_name = None
    for p in preferred:
        if p in backend_names:
            best_name = p
            break
    if best_name is None:
        best_name = backend_names[0]

    with open("ibm_backend.txt", "w") as f:
        f.write(best_name)
    with open("ibm_credentials.txt", "w") as f:
        f.write(f"api_key={IBM_API_KEY}\ncrn={IBM_CRN}\nbackend={best_name}\n")

    print(f"""
  ╔══════════════════════════════════════════════════╗
  ║  CONNECTED SUCCESSFULLY                          ║
  ║  Selected backend : {best_name:<28}║
  ║  All backends     : {str(backend_names)[:28]:<28}║
  ║  Credentials saved: ibm_credentials.txt         ║
  ║  Backend saved    : ibm_backend.txt             ║
  ╚══════════════════════════════════════════════════╝

  Setup complete!
  Next step: python -m qgan.train_ibm
    """)

except Exception as e:
    print(f"  Error: {e}")
    print("  Share this output")
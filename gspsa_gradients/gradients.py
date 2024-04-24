import importlib.util

# Check if TFQ is installed
tfq_spec = importlib.util.find_spec("tensorflow_quantum")
tfq_installed = tfq_spec is not None

# Check if Qiskit is installed
qiskit_spec = importlib.util.find_spec("qiskit")
qiskit_installed = qiskit_spec is not None

if tfq_installed:
    pass
elif qiskit_installed:
    pass
else:
    raise ImportError("Neither TFQ nor Qiskit is installed.")
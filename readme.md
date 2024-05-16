# Guide-SPSA Gradients

This repository implements the Guided-SPSA gradients introduced in "[Guided-SPSA: Simultaneous Perturbation Stochastic Approximation assisted by the Parameter Shift Rule by M. Periyasamy et. al.](https://arxiv.org/abs/2404.15751)"

## Features

- **Integration:** Easily integrate with Qiskit and Tensorflow-Quantum APIs. The code inherits the respective differentiators provided by the APIs. Hence, one can use this implementation in any standard machine learning work that deals with gradient estimation via "expectation values" in Qiskit or TensorFlow-Quanutm.

## Setup and Installation

The library requires an installation of `python > 3.9 `, and following libraries:
- Qiskit version
    - `qiskit`
    - `qiskit-algorithms`
- Tensorflow-Quantum version
    - `tensorflow`
    - `tesnroflow-quantum`

The package can be installed via pip:
- package without other dependencies
    ```bash
    pip install gspsa-gradients
    ```

- To install the with qiskit dependency:

    ```bash
    pip install gspsa-gradients[qiskit]
    ```
- To install the with tfq dependency:

    ```bash
    pip install gspsa-gradients[tfq]
    ```

- The package `gspsa-gradients` can be installed locally via:
    ```
    git clone https://github.com/maniraman-periyasamy/gspsa-gradients.git
    cd gspsa-gradients
    pip install -e .
    ```

## Usage and Examples

- Qiskit version
    ```python
    
    from gspsa_gradients.qiskit_gradient import GSPSAEstimatorGradient
    from qiskit.primitives import Estimator
    gradient = GSPSAEstimatorGradient(total_steps = num_epochs, estimator=Estimator(), num_observables = len(observables), 
        tau=0.5, spsa_epsilon=0.01, damping_coeff=0.5)
    qnn1 = EstimatorQNN(
        circuit=qc, input_params=inputParams, weight_params=circuitParams, observables=observables, gradient=gradient, input_gradients=False)

    ...
    
    ```
- TFQ verison
    ```python

    from gspsa_gradients.tfq_gradient import GSPSAGradient
    diff=GSPSAGradient(total_steps=num_epochs, input_dims=num_inputs, tau=0.5, spsa_epsilon=0.01, damping_coeff=0.5)
    quantum_layer = tfq.layers.ControlledPQC(circuit, observables, differentiator=diff, repetitions=1024) # make repitions to 0 for exact expectation value estimation 

    ...
    ```

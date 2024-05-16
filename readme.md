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

The package `gspsa-gradients` can be installed locally via:
```
git clone https://github.com/maniraman-periyasamy/gspsa-gradients.git
cd gspsa-gradients
pip install -e .
```

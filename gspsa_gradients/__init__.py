"""
GSPSA-Gradients

An implementation of Guided-SPSA gradient estimation technique introduced here: https://arxiv.org/abs/2404.15751
"""

__version__ = "0.1.0"
__author__ = 'Maniraman Periyasay'


name='gspsa_gradients'

from . import qiskit_gradient
from . import tfq_gradient
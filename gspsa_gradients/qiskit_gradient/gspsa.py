# Author : Maniraman Periyasamy
# This code is part of gspsa gradients repository.
# This code uses parts of code snippets from qiskit
# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from __future__ import annotations

from collections.abc import Sequence

from typing import (
    Any,
    Dict,
    Union,
    Optional,
    Sequence
)

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers import Options
from qiskit.primitives import BaseEstimator

from qiskit_algorithms.gradients import BaseEstimatorGradient 
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_algorithms.gradients import SPSAEstimatorGradient 
from qiskit_algorithms.gradients import EstimatorGradientResult
from qiskit_algorithms.gradients.utils import DerivativeType

import numpy as np

class GSPSAEstimatorGradient(BaseEstimatorGradient):
    """
    Compute the gradients of the expectation values by the Guided-SPSA method [1].

    **Reference:**
    [1] _
    """
    def __init__(self, estimator: BaseEstimator,   total_steps: int, num_observables: int,
        k_max:int | None = None, k_min: int | None =None, tau:float = 0.5, damping_coeff:float = 1.0, spsa_epsilon:float = 1.0,
        options: Options | None = None, derivative_type: DerivativeType = DerivativeType.REAL):
        super().__init__(estimator, options, derivative_type)

        self._steps = total_steps
        self._num_observables = num_observables
        self._kMax = k_max
        self._kMin = k_min
        self._k = k_min
        self._stepCounter = 0
        self._gamma = None
        self._tau = tau
        assert 0.0 <= self._tau <= 1.0, f"tau = {self._tau} is not within the expected range [0,1]"
        self._epsilon = damping_coeff # This is the epsilon for spsa damping in guided spsa!!!
        if self._kMin != None and self._kMax != None:
            self._gamma = (self._kMax - self._kMin)/self._steps
        
        self._paramShift = ParamShiftEstimatorGradient(estimator=estimator, options=options, derivative_type=derivative_type)
        self._spsa = SPSAEstimatorGradient(estimator=estimator, options=options, epsilon=spsa_epsilon)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> EstimatorGradientResult:
        
        if self._kMin == None or self._kMax == None:
            if self._kMin == None:
                self._kMin = max(1, len(parameters[0])*0.1)
            if self._kMax == None:
                self._kMax = len(parameters[0])*min(1, 1.5-self._tau)
            self._gamma = (self._kMax - self._kMin)/self._steps

        self._k = int(min(self._kMin + self._stepCounter*self._gamma, self._kMax))
        self._spsa._batch_size = self._k
        g_circuits, g_parameter_values, g_parameters = self._preprocess(
            circuits, parameter_values, parameters, self._paramShift.SUPPORTED_GATES
        )
        results = self._run_unique(
            g_circuits, observables, g_parameter_values, g_parameters, **options
        )
        return results

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        inputLength = int(len(circuits)/self._num_observables)
        parameterShiftSamples = int(inputLength*self._tau)
        spsaSamples = inputLength - parameterShiftSamples
        
        # split ParameterShift Samples
        circuitsPS = []
        observablesPS = []
        parameter_valuesPS = []
        parametersPS = []
        for i in range(self._num_observables):
            start_idx = i*inputLength
            end_idx = i*inputLength+parameterShiftSamples
            circuitsPS += circuits[start_idx : end_idx]
            observablesPS += observables[start_idx : end_idx]
            parameter_valuesPS += parameter_values[start_idx : end_idx]
            parametersPS += parameters[start_idx : end_idx]

        # split SPSA Samples
        circuitsSPSA = []
        observablesSPSA = []
        parameter_valuesSPSA = []
        parametersSPSA = []
        for i in range(self._num_observables):
            start_idx = i*inputLength+parameterShiftSamples
            end_idx = i*inputLength+inputLength
            circuitsSPSA += circuits[start_idx : end_idx]
            observablesSPSA += observables[start_idx : end_idx]
            parameter_valuesSPSA += parameter_values[start_idx : end_idx]
            parametersSPSA += parameters[start_idx : end_idx]
        psJob = self._paramShift.run(circuitsPS, observablesPS, parameter_valuesPS, parametersPS, **options)
        spsaJob = self._spsa.run(circuitsSPSA, observablesSPSA, parameter_valuesSPSA, parametersSPSA, **options)
        jacobianPS = psJob.result()
        jacobianSPSA = spsaJob.result()
        
        
        grad = np.zeros((inputLength, self._num_observables, len(parameters[0])))
        metadata = []
        for i in range(self._num_observables):
            grad[:parameterShiftSamples, i, :] = jacobianPS.gradients[i * parameterShiftSamples: (i + 1) * parameterShiftSamples]
            grad[parameterShiftSamples:, i, :] = jacobianSPSA.gradients[i * spsaSamples: (i + 1) * spsaSamples]
            metadata += jacobianPS.metadata[i * parameterShiftSamples: (i + 1) * parameterShiftSamples]
            metadata += jacobianSPSA.metadata[i * spsaSamples: (i + 1) * spsaSamples]


        grad_norm = np.linalg.norm(grad, ord=2, axis=2)
        spsa_norm_suppresent = np.mean(grad_norm[:parameterShiftSamples], axis=0)/grad_norm[parameterShiftSamples:]
        spsa_norm_suppresent = spsa_norm_suppresent*self._epsilon
        mult_factor = np.where(spsa_norm_suppresent<=1.0, spsa_norm_suppresent, 1.0)

        grad[parameterShiftSamples:] = np.multiply(grad[parameterShiftSamples:], mult_factor[:, :, np.newaxis])
        grad = grad.transpose(1,0,2).reshape(inputLength*self._num_observables, len(parameters[0])) # bring it to orginal shape grouped by obervables
        grad = [row for row in grad] # convert to list of 1D arrays
        
        return EstimatorGradientResult(gradients=grad, metadata=metadata, options=jacobianPS.options)
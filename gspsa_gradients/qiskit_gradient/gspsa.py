# This code is standalone.
#
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



class GSPSAEstimatorGradient(BaseEstimatorGradient):
    """
    Compute the gradients of the expectation values by the Guided-SPSA method [1].

    **Reference:**
    [1] _
    """
    def __init__(self, estimator: BaseEstimator, steps: int,  
        k_max:int | None = None, k_min: int | None =None, tau:float = 0.5, epsilon:float = 1.0,
        options: Options | None = None, derivative_type: DerivativeType = DerivativeType.REAL):
        super().__init__(estimator, options, derivative_type)

        self._steps = steps
        self._kMax = k_max
        self._kMin = k_min
        self._stepCounter = 0
        self._gamma = None
        self._tau = tau
        self._epsilon = epsilon
        if self._kMin != None and self._kMax != None:
            self._gamma = (self._kMax - self._kMin)/self._steps
        
        self._paramShift = ParamShiftEstimatorGradient(estimator=estimator, options=options, derivative_type=derivative_type)
        self._spsa = SPSAEstimatorGradient(estimator=estimator, options=options, derivative_type=derivative_type)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> EstimatorGradientResult:
        
        g_circuits, g_parameter_values, g_parameters = self._preprocess(
            circuits, parameter_values, parameters, self._paramShift.SUPPORTED_GATES
        )
        results = self._run_unique(
            g_circuits, observables, g_parameter_values, g_parameters, **options
        )
        return self._postprocess(results, circuits, parameter_values, parameters)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        
        pass
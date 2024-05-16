# Author : Maniraman Periyasamy
# This code is part of gspsa gradients repository.
# This code uses parts of code snippets from tensorflow and tesnroflow-quantum
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

import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow_quantum.python.differentiators import differentiator, ParameterShift
import math

from collections.abc import Sequence

from typing import (
    Any,
    Dict,
    Union,
    Optional,
    Sequence
)

class GSPSAGradient(differentiator.Differentiator):

    def __init__(self, total_steps: int, input_dims: int,
        k_max:int | None = None, k_min: int | None =None, tau:float = 0.5, damping_coeff:float = 1.0, spsa_epsilon:float = 1.0) -> None:
        super().__init__()

        self._steps = total_steps
        self._inputDims = input_dims
        self._kMax = k_max
        self._kMin = k_min
        self._k = k_min
        self._stepCounter = 0
        self._gamma = None
        self._tau = tau
        assert 0.0 <= self._tau <= 1.0, f"tau = {self._tau} is not within the expected range [0,1]"
        self._epsilon = damping_coeff # This is the epsilon for spsa damping in guided spsa!!!
        self._delta = spsa_epsilon
        if self._kMin != None and self._kMax != None:
            self._gamma = (self._kMax - self._kMin)/self._steps



        """self.batch_size = batch_size
        self.delta = delta
        self.const_suppress = const_suppressant
        self.batch_slope = (40-4)/50000
        self.counter = 0
        self.g_spsa_param_ratio = g_spsa_param_ratio
        self.weights_length=weights_length
        self.inputs_length=inputs_length
        self.grads_length = self.weights_length+self.inputs_length"""
        
        
    def generate_differentiable_op(self, *, sampled_op=None, analytic_op=None):
        
        if sampled_op is not None:
            return super().generate_differentiable_op(sampled_op=sampled_op)

        return super().generate_differentiable_op(analytic_op=analytic_op)

    @tf.function
    def get_gradient_circuits(self, programs, symbol_names, symbol_values):
        """See base class description."""
        raise NotImplementedError(
            "This method is not needed as the gradient circuits are are generated internally with Parameter-shift rule")

    @differentiator.catch_empty_inputs
    @tf.function
    def differentiate_analytic(self, programs, symbol_names, symbol_values,
                               pauli_sums, forward_pass_vals, grad):
        


        if self._kMin == None or self._kMax == None:
            if self._kMin == None:
                self._kMin = max(1, len(symbol_names)*0.1)
            if self._kMax == None:
                self._kMax = len(symbol_names)*min(1, 1.5-self._tau)
            self._gamma = (self._kMax - self._kMin)/self._steps

        self._k = int(min(self._kMin + self._stepCounter*self._gamma, self._kMax))

        parameterShiftSamples = int(len(programs)*self._tau)
        spsaSamples = len(programs) - parameterShiftSamples

        ps = ParameterShift()
        expectation_op = tfq.get_expectation_op()
        programs_ps = programs[:parameterShiftSamples]
        symbol_values_ps = symbol_values[:parameterShiftSamples, :]
        pauli_sums_ps = pauli_sums[:parameterShiftSamples, :]
        batch_programs, new_symbol_names, batch_symbol_values, batch_weights, batch_mapper = ps.get_gradient_circuits(programs = programs_ps, symbol_names=symbol_names, symbol_values=symbol_values_ps)
        #ps.differentiate_analytic(programs, symbol_names, symbol_values,
        #                       pauli_sums, forward_pass_vals, grad)
        
        m_i = tf.shape(batch_programs)[1]
        batch_pauli_sums = tf.tile(tf.expand_dims(pauli_sums_ps, 1), [1, m_i, 1])
        n_batch_programs = tf.reduce_prod(tf.shape(batch_programs))
        n_symbols = tf.shape(new_symbol_names)[0]
        n_ops = tf.shape(pauli_sums_ps)[1]
        batch_expectations = expectation_op(
            tf.reshape(batch_programs, [n_batch_programs]), new_symbol_names,
            tf.reshape(batch_symbol_values, [n_batch_programs, n_symbols]),
            tf.reshape(batch_pauli_sums, [n_batch_programs, n_ops]))
        batch_expectations = tf.reshape(batch_expectations,
                                        tf.shape(batch_pauli_sums))

        # has shape [n_programs, n_symbols, n_ops]
        ps_jacobian = tf.map_fn(
            lambda x: tf.einsum('sm,smo->so', x[0], tf.gather(x[1], x[2])),
            (batch_weights, batch_expectations, batch_mapper),
            fn_output_signature=tf.float32)
        
        
        programs_spsa = programs[parameterShiftSamples:]
        symbol_values_spsa = symbol_values[parameterShiftSamples:, :]
        pauli_sums_spsa = pauli_sums[parameterShiftSamples:, :]
        
        spsa_jacobian = tf.zeros((parameterShiftSamples,len(symbol_names),2))
        for i in range(self._k):
            delta_shift = tf.cast(
                    2 * tf.random.uniform(shape=(spsaSamples, len(symbol_names) - self._inputDims),
                                        minval=0,
                                        maxval=2,
                                        dtype=tf.int32) - 1, tf.float32)
            delta_shift = tf.concat([tf.zeros((spsaSamples, self._inputDims)), delta_shift], axis=1)
            symbol_values_spsa_positive = symbol_values_spsa + self._delta*delta_shift
            symbol_values_spsa_negative = symbol_values_spsa - self._delta*delta_shift 
            spsa_exp_positve = expectation_op(programs=programs_spsa, symbol_names=symbol_names, symbol_values=symbol_values_spsa_positive, pauli_sums=pauli_sums_spsa)
            spsa_exp_negative = expectation_op(programs=programs_spsa, symbol_names=symbol_names, symbol_values=symbol_values_spsa_negative, pauli_sums=pauli_sums_spsa)
            spsa_diff = ((spsa_exp_positve - spsa_exp_negative)/(2*self._delta))
            #spsa_grad = tf.tensordot(delta_shift, spsa_diff, axes=[0,0])
            delta_shift = tf.expand_dims(delta_shift, axis=-1)
            spsa_diff = tf.expand_dims(spsa_diff, axis=1)
            spsa_grad = tf.multiply(delta_shift, spsa_diff)
            spsa_jacobian += spsa_grad
        spsa_jacobian /= self._k
        pass
        
        jacobian = tf.concat([ps_jacobian, spsa_jacobian], axis=0) 
        jacobian_norms = tf.norm(jacobian, ord=2, axis=1)
        ps_jacobian_avg_norm = tf.math.reduce_mean(jacobian_norms[:parameterShiftSamples], axis=0)
        spsa_norm_suppresent = ps_jacobian_avg_norm/jacobian_norms[parameterShiftSamples:]
        spsa_norm_suppresent = spsa_norm_suppresent*self._epsilon
        spsa_norm_suppresent = tf.where(spsa_norm_suppresent<1.0, spsa_norm_suppresent, 1.0)
        spsa_jacobian = tf.multiply(spsa_jacobian, tf.expand_dims(spsa_norm_suppresent, axis=1))
        jacobian = tf.concat([ps_jacobian, spsa_jacobian], axis=0)
        
        grads_chain = tf.einsum('pso,po->ps', jacobian, grad)
        self._stepCounter +=1
        
        return grads_chain

    def differentiate_sampled(self, programs, symbol_names, symbol_values,
                              pauli_sums, num_samples, forward_pass_vals, grad):
        if self._kMin == None or self._kMax == None:
            if self._kMin == None:
                self._kMin = max(1, len(symbol_names)*0.1)
            if self._kMax == None:
                self._kMax = len(symbol_names)*min(1, 1.5-self._tau)
            self._gamma = (self._kMax - self._kMin)/self._steps

        self._k = int(min(self._kMin + self._stepCounter*self._gamma, self._kMax))

        parameterShiftSamples = int(len(programs)*self._tau)
        spsaSamples = len(programs) - parameterShiftSamples

        ps = ParameterShift()
        expectation_op = tfq.get_sampled_expectation_op()
        programs_ps = programs[:parameterShiftSamples]
        symbol_values_ps = symbol_values[:parameterShiftSamples, :]
        pauli_sums_ps = pauli_sums[:parameterShiftSamples, :]
        num_samples_ps = num_samples[:parameterShiftSamples, :]
        batch_programs, new_symbol_names, batch_symbol_values, batch_weights, batch_mapper = ps.get_gradient_circuits(programs = programs_ps, symbol_names=symbol_names, symbol_values=symbol_values_ps)
       
        m_i = tf.shape(batch_programs)[1]
        batch_pauli_sums = tf.tile(tf.expand_dims(pauli_sums_ps, 1), [1, m_i, 1])
        batch_num_samples = tf.tile(tf.expand_dims(num_samples_ps, 1), [1, m_i, 1])
        n_batch_programs = tf.reduce_prod(tf.shape(batch_programs))
        n_symbols = tf.shape(new_symbol_names)[0]
        n_ops = tf.shape(pauli_sums_ps)[1]
        batch_expectations = expectation_op(
            tf.reshape(batch_programs, [n_batch_programs]), new_symbol_names,
            tf.reshape(batch_symbol_values, [n_batch_programs, n_symbols]),
            tf.reshape(batch_pauli_sums, [n_batch_programs, n_ops]),
            tf.reshape(batch_num_samples, [n_batch_programs, n_ops]))
        batch_expectations = tf.reshape(batch_expectations,
                                        tf.shape(batch_pauli_sums))

        # has shape [n_programs, n_symbols, n_ops]
        ps_jacobian = tf.map_fn(
            lambda x: tf.einsum('sm,smo->so', x[0], tf.gather(x[1], x[2])),
            (batch_weights, batch_expectations, batch_mapper),
            fn_output_signature=tf.float32)
        
        
        programs_spsa = programs[parameterShiftSamples:]
        symbol_values_spsa = symbol_values[parameterShiftSamples:, :]
        pauli_sums_spsa = pauli_sums[parameterShiftSamples:, :]
        num_samples_spsa = num_samples[parameterShiftSamples:, :]
        
        spsa_jacobian = tf.zeros((parameterShiftSamples,len(symbol_names),2))

        for i in range(self._k):
            delta_shift = tf.cast(
                    2 * tf.random.uniform(shape=(spsaSamples, len(symbol_names) - self._inputDims),
                                        minval=0,
                                        maxval=2,
                                        dtype=tf.int32) - 1, tf.float32)
            delta_shift = tf.concat([tf.zeros((spsaSamples, self._inputDims)), delta_shift], axis=1)
            symbol_values_spsa_positive = symbol_values_spsa + self._delta*delta_shift
            symbol_values_spsa_negative = symbol_values_spsa - self._delta*delta_shift 
            spsa_exp_positve = expectation_op(programs=programs_spsa, symbol_names=symbol_names, symbol_values=symbol_values_spsa_positive, pauli_sums=pauli_sums_spsa, num_samples=num_samples_spsa)
            spsa_exp_negative = expectation_op(programs=programs_spsa, symbol_names=symbol_names, symbol_values=symbol_values_spsa_negative, pauli_sums=pauli_sums_spsa, num_samples=num_samples_spsa)
            spsa_diff = ((spsa_exp_positve - spsa_exp_negative)/(2*self._delta))
            
            delta_shift = tf.expand_dims(delta_shift, axis=-1)
            spsa_diff = tf.expand_dims(spsa_diff, axis=1)
            spsa_grad = tf.multiply(delta_shift, spsa_diff)
            spsa_jacobian += spsa_grad
        spsa_jacobian /= self._k
        
        jacobian = tf.concat([ps_jacobian, spsa_jacobian], axis=0) 
        jacobian_norms = tf.norm(jacobian, ord=2, axis=1)
        ps_jacobian_avg_norm = tf.math.reduce_mean(jacobian_norms[:parameterShiftSamples], axis=0)
        spsa_norm_suppresent = ps_jacobian_avg_norm/jacobian_norms[parameterShiftSamples:]
        spsa_norm_suppresent = spsa_norm_suppresent*self._epsilon
        spsa_norm_suppresent = tf.where(spsa_norm_suppresent<1.0, spsa_norm_suppresent, 1.0)
        spsa_jacobian = tf.multiply(spsa_jacobian, tf.expand_dims(spsa_norm_suppresent, axis=1))
        jacobian = tf.concat([ps_jacobian, spsa_jacobian], axis=0)
        
        grads_chain = tf.einsum('pso,po->ps', jacobian, grad)
        self._stepCounter +=1
        
        return grads_chain
'''
Created: Sprint 2
Last Modified: Sprint 2

Purpose:
'''

import cupy as cp

from neural.core.base import Activation
from tools.debugging import debug_style


class ReLU(Activation):
    '''
    Clips outputs below 0.
    Formula: max(x, 0)
    '''

    def fwd(self, inputs: cp.ndarray) -> None:

        self.inputs = inputs

        self.outputs = cp.maximum(inputs, 0, dtype='float32')
    
    def bwd(self, chain_dvals: cp.ndarray) -> None:
        
        # Derivative is 1 if inputs >= 0, 0 otherwise
        # True/False arr -> 1s & 0s
        local_input_dvals = (self.inputs != 0).astype('float32')

        # Chain rule, element-wise as they are the same shape
        self.input_dvals = chain_dvals * local_input_dvals


class Sigmoid(Activation):
    '''
    Squashes inputs to between 0 & 1.
    Formula: 1 / (1 + e^-x)
    '''


    def fwd(self, inputs: cp.ndarray) -> None:

        self.inputs = inputs

        exp_vals = cp.exp(inputs)
        self.outputs = 1 / (1 + exp_vals)
    

class Tanh(Activation):
    '''
    Squashes inputs to between -1 and 1.
    Formula: (e^x - e^-x) / (e^x + e^-x)
    '''

    def fwd(self, inputs: cp.ndarray) -> None:
        
        self.inputs = inputs
        
        pos_exp_vals = cp.exp(inputs)
        neg_exp_vals = cp.exp(-inputs)

        self.outputs = (pos_exp_vals - neg_exp_vals) / (pos_exp_vals + neg_exp_vals)


class ReLULeaky(Activation):
    '''
    Makes values below 0 small using a leak rate.
    Formula: max(x, leak_rate*x)
    '''


    def __init__(self, leak_rate: float) -> None:
        '''Define leak rate (between 0 & 1) for outputs < 0.'''
    
        if leak_rate < 0:
            raise ValueError('The leak rate must be between 0 & 1!')

        super().__init__()

        self.leak_rate = leak_rate

    def fwd(self, inputs: cp.ndarray) -> None: # type: ignore
        
        self.inputs = inputs

        # leak_rate * inputs is only greater in the negative quadrant
        small_outputs = self.leak_rate * inputs
        self.outputs = cp.maximum(small_outputs, inputs)


class Softmax(Activation):
    '''
    Convert raw network outputs into a probability distribution.
    '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.one_overflows = 0
        self.debug = False


    def fwd(self, inputs: cp.ndarray) -> None: # type: ignore 
        '''
        Sum exponentiated inputs and divide each exponentiated input by this sum.
        '''        

        self.inputs = inputs

        # print(debug_style(f'Inputs to Softmax: {inputs}'))

        # Shifting all inputs by same value has no effect on output distribution
        max_input = cp.max(inputs, axis=1, keepdims=True) # axis 1 = per sample
        exp_vals = cp.exp(inputs - max_input, dtype='float32')

        # axis 0 = batch-wise, axis 1 = neuron-wise
        exp_sum = cp.sum(exp_vals, axis=1, keepdims=True, dtype='float32') 

        self.outputs = exp_vals / exp_sum

        # Print debug info if any neuron dominates the output
        prints = False
        if self.debug:
            for i, sample in enumerate(self.outputs):
                if cp.any(sample == 1):
                    if prints:
                        print(f'1 detected in Softmax outputs:\n{self.outputs[i]}.\n'
                            f'exp_sum:{exp_sum[i]}\nexp_vals:{exp_vals[i]}\n'
                            f'inputs:{inputs[i]}')

                    self.one_overflows += 1
    

    def bwd(self, chain_dvals: cp.ndarray, true_classes: cp.ndarray) -> None: # pyright: ignore[reportIncompatibleMethodOverride]
        '''
        Calculate the derivatives of Softmax w.r.t inputs, using chain rule.

        :param: chain_dvals -- The derivatives of Loss w.r.t Softmax outputs.
        '''

        # si = softmax output for neuron at index i
        # sj = softmax output for neuron at index of true class (1)
        
        # sj_dvals:
        # sj * (1 - sj) * chain_dvals
        # (sj * chain) - sj * (sj * chain)

        # for si, dval = -si * sj * chain_dvals

        # chain_dvals: zeros at i, -1/sj for j 
        # chain_dvals @ (1- inputs): derivatives for sj, 0 otherwise

        # could do:
        # for sj: chain_dvals @ (1 - inputs)
        # for si: - si * inverted chain_dvals (elementwise)

        sj_chain_product = chain_dvals * self.inputs

        sj_dvals = chain_dvals - sj_chain_product
        
        si_dvals = self.inputs

        print(chain_dvals.shape, self.inputs.shape)
        exit()
        


__all__ = ["ReLU", "Sigmoid", "Tanh", "ReLULeaky", "Softmax"]



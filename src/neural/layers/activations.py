'''
Created: Sprint 2
Last Modified: Sprint 2

Purpose:
'''

import numpy as np 

from neural.core.base import Activation
from tools.debugging import debug_style


class ReLU(Activation):
    '''
    Clips outputs below 0.
    Formula: max(x, 0)
    '''

    @staticmethod
    def _fwd_func(inputs: np.ndarray) -> np.ndarray:
        
        outputs = np.maximum(inputs, 0)
        return outputs


class Sigmoid(Activation):
    '''
    Squashes inputs to between 0 & 1.
    Formula: 1 / (1 + e^-x)
    '''

    @staticmethod
    def _fwd_func(inputs: np.ndarray) -> np.ndarray:        

        exp_vals = np.exp(inputs)
        outputs = 1 / (1 + exp_vals)
        
        return outputs
    

class Tanh(Activation):
    '''
    Squashes inputs to between -1 and 1.
    Formula: (e^x - e^-x) / (e^x + e^-x)
    '''

    @staticmethod
    def _fwd_func(inputs: np.ndarray) -> np.ndarray:
        
        pos_exp_vals = np.exp(inputs)
        neg_exp_vals = np.exp(-inputs)

        outputs = (pos_exp_vals - neg_exp_vals) / (pos_exp_vals + neg_exp_vals)

        return outputs


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


    def _fwd_func(self, inputs: np.ndarray) -> np.ndarray: # type: ignore
        
        # leak_rate * inputs is only greater in the negative quadrant
        small_outputs = self.leak_rate * inputs
        outputs = np.maximum(small_outputs, inputs)

        return outputs


class Softmax(Activation):
    '''
    Convert raw network outputs into a probability distribution.
    '''

    @staticmethod
    def _fwd_func(inputs: np.ndarray) -> np.ndarray: # type: ignore 
        '''
        Sum exponentiated inputs and divide each exponentiated input by this sum.
        '''        

        exp_vals = np.exp(inputs)

        # axis 0 = batch-wise, axis 1 = neuron-wise
        exp_sum = np.sum(exp_vals, axis=1, keepdims=True) 

        outputs = exp_vals / exp_sum

        return outputs


def leaky_relu_test():
    
    print(debug_style('Leaky ReLU Test'))
    activ_func = ReLULeaky(leak_rate=0.1)

    # Mix of pos/neg floats
    test_data = np.random.randn(4, 3)
    print(f'test data: {test_data}\n')
    
    # (Should be) floats between 0 & 1
    activ_func.fwd(test_data)
    print(f'outputs: {activ_func.outputs}\n')

def relu_test():
    print(debug_style('ReLU Test'))
    activ_func = ReLU()

    # Mix of pos/neg floats
    test_data = np.random.randn(4, 3)
    print(f'test data: {test_data}\n')
    
    # (Should be) floats between 0 & 1
    activ_func.fwd(test_data)
    print(f'outputs: {activ_func.outputs}\n')



__all__ = ["ReLU", "Sigmoid", "Tanh", "ReLULeaky", "Softmax"]


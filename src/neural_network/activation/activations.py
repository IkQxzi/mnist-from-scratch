'''
Created: Sprint 2
Last Modified: Sprint 2

Purpose:
'''

import numpy as np 

from tools.debugging import debug_style

class Activation():
    '''A generic activation class for all activation functions.'''

    def __init_subclass__(cls) -> None:
        
        # Calls the __init_subclass__ for all parents
        # to prevent overriding parent class behaviour
        super().__init_subclass__()

        # All activation functions require forwarding logic for fwd() to work
        if '_fwd_func' not in cls.__dict__:
            print(f'WARNING: No fwd() method implemented for layer type: {cls.__name__}')

        # Check for bwd() later

    @staticmethod
    def _fwd_func(inputs: np.ndarray) -> np.ndarray: # type: ignore
        '''
        The forwarding logic calculation.
        
        Keyword Args:
            inputs -- the inputs to the activation function (output from dense layer)

        '''
        pass

    def fwd(self, inputs: np.ndarray) -> None:
        '''
        Performs forward pass using _fwd_func().

        Attributes:
            outputs -- the outputs of the forward pass
        '''

        self.outputs = self._fwd_func(inputs)

    def bwd(self) -> None:
        '''Calculates the derivative of each layer.'''
        
        pass


class ReLU(Activation):
    '''Clips outputs below 0.'''

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
        
        exp_vals = np.exp(-inputs)
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


def tanh_test():
    relu = Tanh()

    # Mix of pos/neg floats
    test_data = np.random.randn(4, 3)
    print(f'test data: {test_data}\n')
    
    # (Should be) floats between -1 & 1
    relu.fwd(test_data)
    print(f'outputs: {relu.outputs}\n')

tanh_test()
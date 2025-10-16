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

relu = ReLU()
test_data = np.random.randn(4, 3)
print(test_data)
relu.fwd(test_data)
print(relu.outputs)


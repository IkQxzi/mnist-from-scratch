from typing import List, Dict, Union, Optional

import numpy as np

from tools.debugging import warning_style

class LayerGeneric:
    '''A generic structure which will be used for both core layers and activation functions.'''

    def __init_subclass__(cls) -> None:
        
        # Calls the __init_subclass__ for all parents (good practice)
        # to prevent overriding parent class behaviour
        super().__init_subclass__()

        # Useful logging for child classes (layers):

        if 'fwd' not in cls.__dict__:
            print(f'WARNING: No fwd() method implemented for layer type: {cls.__name__}')
        
        if 'bwd' not in cls.__dict__:
            print(f'WARNING: No bwd() method implemented for layer type: {cls.__name__}')


    def __init__(self) -> None:
        pass


    def fwd(self, inputs: np.ndarray):
        pass

    def bwd(self, dvalues: np.ndarray):
        pass


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


class Loss():

    def __init_subclass__(cls) -> None:
        
        # Calls the __init_subclass__ for all parents
        # to prevent overriding parent class behaviour
        super().__init_subclass__()

        # All loss functions require forwarding logic for fwd() to work
        if '_fwd_func' not in cls.__dict__:
            print(f'WARNING: No fwd() method implemented for layer type: {cls.__name__}')

        # Check for bwd() later


    @staticmethod
    def _fwd_func(pred_classes: np.ndarray, true_classes: np.ndarray) -> np.ndarray: # type: ignore
        '''
        The loss calculation.
        
        Keyword Args:
            pred_classes -- predicted classes from the final layer
            true_classes -- indexes of the correct/intended outputs
            
        '''
        pass


    def fwd(self, pred_classes: np.ndarray, true_classes: np.ndarray) -> None:
        '''
        Performs forward pass using _fwd_func().

        Attributes:
            outputs -- the outputs of the forward pass
        '''

        self.outputs = self._fwd_func(pred_classes, true_classes)


    def bwd(self) -> None:
        '''Calculates the derivative of each layer.'''
        
        pass
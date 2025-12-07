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
        if 'fwd' not in cls.__dict__:
            print(f'WARNING: No fwd() method implemented for layer type: {cls.__name__}')

        if 'bwd' not in cls.__dict__:
            print(f'WARNING: No bwd() method implemented for layer type: {cls.__name__}')


    def fwd(self, inputs: np.ndarray) -> None:
        '''
        Performs forward pass.

        Attributes:
            outputs -- the outputs of the forward pass.
        '''

        self.outputs: np.ndarray
        pass


    def bwd(self, chain_dvals: np.ndarray) -> None: # type: ignore
        '''
        Performs forward pass using _fwd_func().

        Attributes:
            input_dvals -- The derivatives of the output w.r.t inputs.
        '''
        self.input_dvals: np.ndarray
        pass


# class ActivationOutput(Activation):

#     def bwd(self, chain_dvals: np.ndarray, true_classes: np.ndarray) -> None: # pyright: ignore[reportIncompatibleMethodOverride]

#         pass


class Loss():

    def __init_subclass__(cls) -> None:
        
        # Calls the __init_subclass__ for all parents
        # to prevent overriding parent class behaviour
        super().__init_subclass__()

        # All loss functions require forwarding logic for fwd() to work
        if 'calculate' not in cls.__dict__:
            print(f'WARNING: No calculate() method implemented for layer type: {cls.__name__}')

        if 'bwd' not in cls.__dict__:
            print(f'WARNING: No bwd() method implemented for layer type: {cls.__name__}')


    def calculate(self, pred_classes: np.ndarray, true_classes: np.ndarray) -> np.ndarray: # type: ignore
        '''
        Calculates CCE Loss.

        :param pred_classes: The network's prediction after Softmax pass.
        :param true_classes: The correct classes, one-hot-encoded.
        attributes:
        input_dvals: dvals w.r.t inputs.
        '''

        pass


    def bwd(self) -> None:
        '''Calculates the derivative of loss output w.r.t inputs.
        
        Attributes:
            dvalues -- the derivatives of the layer.
        '''

        pass
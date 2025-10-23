'''
Collection of Loss functions
Created: Sprint 2
Last modified: Sprint 2
'''

import numpy as np



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


class CategoricalCrossEntropy(Loss):
    '''
    Calculate categorical cross-entropy loss using one-hot encoded vectors
    '''

    @staticmethod
    def _fwd_func(pred_classes: np.ndarray, true_classes: np.ndarray) -> np.ndarray: # type: ignore
        '''
        The CCE loss calculation.
        
        Keyword Args:
            pred_classes -- predicted classes from the final layer
            true_classes -- one-hot vectors of the correct/intended outputs
            
        '''

        outputs = np.dot(pred_classes, true_classes)
        return outputs




'''
Collection of Loss functions
Created: Sprint 2
Last modified: Sprint 2
'''

import numpy as np

from neural.core.base import Loss


class CategoricalCrossEntropy(Loss):
    '''
    Calculate categorical cross-entropy loss using one-hot encoded vectors
    '''

    @staticmethod
    def _calc_func(pred_classes: np.ndarray, true_classes: np.ndarray) -> np.ndarray: # type: ignore
        '''
        The CCE loss calculation.
        
        Keyword Args:
            pred_classes -- predicted classes from the final layer
            true_classes -- one-hot vectors of the correct/intended outputs
        '''

        exp_pred_classes = -1 * np.log(pred_classes)
        loss_vals = np.dot(exp_pred_classes, true_classes.T) # for each individual sample
        
        # Taking the sum of all losses
        # which looks like:
        # [[0 0 0 loss 0 0]
        #  [0 loss 0 0 0 0]] -> 2*loss
        loss_sum = np.sum(loss_vals)
        
        # Taking the average loss for the entire batch
        print(loss_sum)
        batch_loss = loss_sum / pred_classes.shape[1]
        return batch_loss

    def bwd(self) -> None:
        pass


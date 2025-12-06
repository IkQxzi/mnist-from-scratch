'''
Collection of Loss functions
Created: Sprint 2
Last modified: Sprint 2
'''

import numpy as np

from neural.core.base import Loss
from tools.debugging.styling import debug_style


class CategoricalCrossEntropy(Loss):
    '''
    Calculate categorical cross-entropy loss using one-hot encoded vectors
    '''

    def calculate(self, pred_classes: np.ndarray, true_classes: np.ndarray) -> np.ndarray: # type: ignore
        '''
        The CCE loss calculation, averaged over a batch of inputs.
        
        Keyword Args:
            pred_classes -- predicted classes from the final layer
            true_classes -- one-hot vectors of the correct/intended outputs
        '''

        self.pred_classes = pred_classes
        self.true_classes = true_classes
        self.batch_size = pred_classes.shape[0]

        # Perform the element wise product of true_classes and pred_classes
        # to get a list of their products (zeros and then relevant values)
        rel_list = true_classes * np.log(pred_classes)
        
        # Then sum the 2D array of relevant predictions
        # to get the relevant predictions for each sample in one value
        total_pred = np.sum(rel_list) 

        # Multiply the sum of all predicted values by -1/N (batch size/# samples)  
        self.outputs = (-1 / self.batch_size) * total_pred


    def bwd(self) -> None:
        '''
        Calculates the derivatives of CCE Loss w.r.t inputs.
        
        :param self: Description
        '''
        
        # - 1 / sj (sj = pj)
        cor_pred_arr = self.true_classes * self.pred_classes
        cor_pred_list = np.sum(cor_pred_arr, axis=1)

        # dvals = -1 / sj (sj = pj)
        local_input_dvals = (-1 * np.ones_like(cor_pred_list)) / cor_pred_list 
        self.input_dval = np.sum(local_input_dvals) / self.batch_size


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
        # to get the relevant predictions for each sample in one list
        total_pred_list = np.sum(rel_list, axis = 1) 

        # Multiply the sum of all predicted values by -1/N (greater loss = worse)
        self.outputs = -1 * total_pred_list


    def bwd(self) -> None:
        '''
        Calculates the derivatives of CCE Loss w.r.t inputs.
        
        :param self: Description
        '''
        
        # - 1 / sj
        cor_pred_arr = self.true_classes * self.pred_classes

        # dvals = -1 / sj
        neg_ones = -1 * np.ones_like(cor_pred_arr)
        zeros = np.zeros_like(cor_pred_arr)

        # Do -1 / sj, skipping div 0 and replacing with 0
        self.input_dvals = np.divide(neg_ones, cor_pred_arr, out=zeros, where=cor_pred_arr!=0)
        


        # self.input_dval = np.sum(local_input_dvals) / self.batch_size


'''
Collection of Loss functions
Created: Sprint 2
Last modified: Sprint 3
'''

import numpy as np

from neural.core.base import Loss

class CategoricalCrossEntropy(Loss):
    '''
    Calculate categorical cross-entropy loss using one-hot encoded vectors
    '''

    def calculate(self, probs: np.ndarray, true_classes: np.ndarray) -> np.ndarray: # type: ignore
        '''
        The CCE loss calculation, averaged over a batch of inputs.
        
        Keyword Args:
            probs -- probability distribution from the final layer
            true_classes -- one-hot vectors of the correct/intended outputs
        '''

        self.probs = probs
        self.true_classes = true_classes
        self.batch_size = probs.shape[0]

        # Perform the element wise product of true_classes and probs
        # to get a list of their products (zeros and then relevant values)
        probs = np.clip(probs, 1e-6, 1)
        # print(f'clipped probs: {probs}')
        rel_list = true_classes * np.log(probs)
        
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
        
        # Get only sj and zeros otherwise
        cor_pred_arr = self.true_classes * self.probs

        # Will be useful to do -1 / sj, and result 0 where sj = 0 
        neg_ones = -1 * np.ones_like(cor_pred_arr)
        zeros = np.zeros_like(cor_pred_arr)

        
        # Change warnings to exceptions, only for div & overflow
        np.seterr(divide='raise', over='raise')

        try:
            # Do -1 / sj, skipping div 0 and replacing with 0
            self.input_dvals = np.divide(neg_ones, cor_pred_arr, out=zeros, where=cor_pred_arr!=0)
        
        # Small div error (not 0 but small)
        except FloatingPointError:
            print(f'neg_ones:\n{neg_ones}\n cor_pred_arr:\n {cor_pred_arr}\n zeros:\n{zeros}')

            # If all zeroes, print true & pred classes
            for i, sample in enumerate(cor_pred_arr):
                if np.all(sample) == 0:
                    print(f'true_classes:\n{self.true_classes[i]}\nprobs:\n{self.probs}')

            exit()
        

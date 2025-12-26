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
        
        # or yj / sj,
        # where sj is prob at index of true class,
        # and yj is true class

        epsilon = 1e-7 # Use epsilon to prevent exploding gradient (from 0 division)
        self.input_dvals = -self.true_classes / (self.probs + epsilon)

        # Get mean dvals to prevent batch sizes from effecting learning
        self.input_dvals /= len(self.true_classes)



'''
Collection of Loss functions
Created: Sprint 2
Last modified: Sprint 3
'''

import cupy as cp

from neural.core.base import Loss

class CCE(Loss):
    '''
    Calculate categorical cross-entropy loss using one-hot encoded vectors
    '''

    def calculate(self, probs: cp.ndarray, true_classes: cp.ndarray) -> cp.ndarray: # type: ignore
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
        probs = cp.clip(probs, 1e-6, 1)
        rel_list = true_classes * cp.log(probs)
        
        # Then sum the 2D array of relevant predictions
        # to get the relevant predictions for each sample in one list
        total_pred_list = cp.sum(rel_list, axis = 1) 

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

        # Use epsilon to prevent exploding gradient (from 0 division)
        # Must be float32 to avoid upcasting to float64
        epsilon = cp.float32(1e-7) 
        self.input_dvals = -self.true_classes / (self.probs + epsilon)

        # Get mean dvals to prevent batch sizes from affecting learning
        # Keep float32 to avoid upcasting to float64
        self.input_dvals /= cp.float32(len(self.true_classes))



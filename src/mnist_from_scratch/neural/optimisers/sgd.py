'''
Stochastic Gradient Descent
Created: Sprint 3
Last Modified: Sprint 3
'''

import cupy as cp

from mnist_from_scratch.tools.debugging import debug_style
from mnist_from_scratch.neural.core.base import Optimiser
from mnist_from_scratch.neural.core.network import NetworkDense


class SGD(Optimiser):

    def __init__(self, network: NetworkDense, learning_rate: float, weight_decay: float) -> None:
        '''
        Define learning rate.
        
        :param: learning_rate -- The fraction of the gradient to update the network's parameters by.
        :param: weight_decay -- The factor by which to consider the squared weights for L2 regularisation.
        '''
        
        self.network = network

        # To keep network params float32
        self.learning_rate = cp.float32(learning_rate)
        self.weight_decay = cp.float32(weight_decay)


    def update_params(self) -> None:

        # Check each layer for Dense (allows for sequential Dense layers)
        for layer in self.network.layers:
            if layer.__class__.__name__ == 'LayerDense':

                # Gradient of total loss w.r.t weights due to CCE Loss
                w_cce_grad = layer.weight_dvals * self.learning_rate

                # Weights decayed due to learning rate and decay rate
                weight_decayed = layer.weights * (1 - 2 * self.learning_rate * self.weight_decay)

                # Update using L2 Regularisation
                layer.weights = weight_decayed - w_cce_grad

                # L2 has no effect on biases
                layer.biases -= (layer.bias_dvals * self.learning_rate)


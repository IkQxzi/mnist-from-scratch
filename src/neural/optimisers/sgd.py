'''
Stochastic Gradient Descent
Created: Sprint 3
Last Modified: Sprint 3
'''

import numpy as np


from tools.debugging import debug_style
from neural.core.base import Optimiser
from neural.core.network import NetworkDense


class StochasticGradientDescent(Optimiser):

    def __init__(self, network: NetworkDense, learning_rate: float, weight_decay: float) -> None:
        '''
        Define learning rate.
        
        :param: learning_rate -- The fraction of the gradient to update the network's parameters by.
        :param: weight_decay -- The factor by which to consider the squared weights for L2 regularisation.
        '''
        
        self.network = network
        self.l_rate = learning_rate
        self.w_decay = weight_decay


    def update_params(self) -> None:

        # Only update Dense layers
        for layer in self.network.layers[::2]:
        
            # Gradient of total loss w.r.t weights due to CCE Loss
            w_cce_grad = layer.weight_dvals * self.l_rate

            # Weights decayed due to learning rate and decay rate
            w_decayed = layer.weights * (1 - 2 * self.l_rate * self.w_decay)

            # Update using L2 Regularisation
            layer.weights = w_decayed - w_cce_grad

            # L2 has no effect on biases
            layer.biases -= (layer.bias_dvals * self.l_rate)


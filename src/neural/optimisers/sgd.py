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

    def __init__(self, network: NetworkDense, learning_rate: float) -> None:
        '''
        Define learning rate.
        
        :param: learning_rate -- The fraction of the gradient to update the network's parameters by.
        '''
        
        self.network = network
        self.l_rate = learning_rate


    def update_params(self) -> None:

        # Only update Dense layers
        for layer in self.network.layers[::2]:
        
            layer.weights -= layer.weight_dvals * self.l_rate
            layer.biases -= layer.bias_dvals * self.l_rate


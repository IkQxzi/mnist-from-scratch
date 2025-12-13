
'''
Created: Sprint 1
Last Modified: Sprint 1
Purpose: Create and propogate neural network
'''

# Built-in imports
import time
from typing import cast, Sequence

# External imports
import numpy as np

# Local imports
# from .base_classes import LayerGeneric
from neural.layers import LayerDense
from neural.core.base import Activation, Network, Layer
from neural.layers.activations import *

from tools.debugging import debug_style, warning_style


class NetworkDense(Network):

    def __init__(self, layers: list) -> None:
        '''
        Initialise a neural network using the specified architecture.

        Keyword Args:
            layers -- The architecture of the network
        
        Attributes:
            layer-arch -- A clean list of the layer names.
        '''

        self.layers = layers # Both dense & activation layers

        # Enforce correct typing so the linter doesn't get mad
        self.layers[::2] = cast(Sequence[LayerDense], self.layers[::2])
        self.layers[1::2] = cast(Sequence[Activation], self.layers[1::2])

        for layer in self.layers:
            print(f'{layer.__class__.__name__}: {type(layer)}')

        # List of layer names (don't know how necessary)
        self.layer_arch: list[str] = []
        for i, arch in enumerate(self.layers):
            if i % 2 == 0: # Indexes of dense layers [0, 2, 4]
                self.layer_arch.append(f'LayerDense{arch.n_neurons, arch.n_inputs}')
            else:
                self.layer_arch.append(f'{arch.__class__.__name__}()')

        print(debug_style(f'Layer arch: {self.layer_arch}'))

        # Will be updated after forwarding
        self.pred_classes = np.array([], dtype=float) 
        self.conf_scores = np.array([], dtype=float)


    def calc_conf_scores(self) -> None:
        '''Calculate confidence scores using self.pred_classes.'''

        output_sum = np.sum(self.pred_classes, axis=1) # Sum of each sample
        self.conf_scores = np.ones_like(self.pred_classes)
        for i in range(output_sum.shape[0]):
            self.conf_scores[i] = (self.pred_classes[i] / output_sum[i])

        self.conf_scores *= 100 # dec to %
        self.conf_scores[self.conf_scores < 0.01] = 0 # Replace small values with 0


    def fwd(self, inputs: np.ndarray):
        '''
        Forward a set of inputs through the entire network.

        Keyword Args:
            inputs -- The inputs to the network
        '''

        # The inputs to the network need be passed only into the first layer
        # not any subsequent ones
        # print(debug_style(f'[NetworkDense.fwd()] Current Layer Index: 0'))
        layer1 = self.layers[0]
        layer1.fwd(inputs)
        print(debug_style(f'[NetworkDense.fwd()] [{layer1.__class__.__name__}] {layer1.outputs.shape}'))

        # For each subsequent layer, forward the outputs of the previous layer as its inputs
        for layer_ind, layer in enumerate(self.layers[1:]):

            
            # print(debug_style(f'[NetworkDense.fwd()] Current Layer Index: {layer_ind+1}'))

            fwd_layer = self.layers[layer_ind] # Uses index of new sliced list
            layer.fwd(fwd_layer.outputs)

            print(debug_style(f'[NetworkDense.fwd()] [{layer.__class__.__name__}] {layer.outputs.shape}'))

        # Update network outputs
        self.pred_classes = self.layers[-1].outputs

        # self.calc_conf_scores()


    def bwd(self, loss_dvals: np.ndarray, true_classes: np.ndarray) -> None:
        '''
        Backpropgate through layers, from Softmax to first layer.
        
        :param: loss_dvals -- Derivative of loss function w.r.t inputs.
        :param: true_classes -- One-hot-encoded vector of correct outputs.
        '''
        
        # Backpropogate through Softmax separately
        print(debug_style(f'Backprop: Loss -> Softmax'))
        output_activ = self.layers[-1]
        output_activ.bwd(chain_dvals=loss_dvals, true_classes=true_classes) # pyright: ignore[reportCallIssue]

        # Number of non-Softmax layers
        n_reg_layers = len(self.layers[:-1])

        # Iterate through layers backwards, from final LayerDense to first LayerDense
        for i, layer in enumerate(self.layers[-2::-1]):
            
            fwd_layer = self.layers[n_reg_layers-i]
            print(debug_style(f'Backprop: {fwd_layer.__class__.__name__} -> {layer.__class__.__name__}'))
            layer.bwd(fwd_layer.input_dvals)  




# layer1 = LayerDense(3, 4)

# test_data = np.array([[0.3, 0.56, -0.3, 0.5],
#                       [5.3, 2.5, -3.5, 2.7],
#                       [3.2, 6.4, 6.2, 0.3]])

'''
Notes: 
- Perhaps move activation functions to other files 
- Move something to other files

# Implementation:
class NetworkDense:
    def __init__(self, architecture):
        self.layers = architecture  # direct list of layers

    def fwd(self, X):
        output = X
        for layer in self.layers:
        
            layer.fwd(output)
            output = layer.output
        self.output = output

# Usage:
network = NetworkDense([
    LayerDense(2, 4),
    ActivationReLU(),
    LayerDense(4, 3),
    ActivationSigmoid()
])
network.fwd(X)


'''
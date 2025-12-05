
'''
Created: Sprint 1
Last Modified: Sprint 1
Purpose: Create and propogate neural network
'''

# Built-in imports
import time
from typing import Dict, Union, Tuple, Sequence

# External imports
import numpy as np

# Local imports
# from .base_classes import LayerGeneric
from neural.layers import LayerDense
from neural.core.base import Activation
from neural.layers.activations import *

from tools.debugging import debug_style, warning_style


class NetworkDense:

    def __init__(self, layers: list[Activation | LayerDense]) -> None:
        '''
        Initialise a neural network using the specified architecture.

        Keyword Args:
            layers -- The architecture of the network
        
        Attributes:
            layer-arch -- A clean list of the layer names.
        '''

        self.layers = layers # Both dense & activation layers
        
        # List of layer names (don't know how necessary)
        self.layer_arch: list[str] = []
        for i, arch in enumerate(self.layers):
            if i % 2 == 0: # Indexes of dense layers [0, 2, 4]
                self.layer_arch.append(f'LayerDense{arch.n_neurons, arch.n_inputs}') # pyright: ignore[reportAttributeAccessIssue]
            else:
                self.layer_arch.append(f'{arch.__class__.__name__}()')

        print(debug_style(f'Layer arch: {self.layer_arch}'))

        self.pred_classes = np.array([], dtype=float) # Will be updated after forwarding

    
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

            prev_layer = self.layers[layer_ind] # Uses index of new sliced list
            layer.fwd(prev_layer.outputs)

            print(debug_style(f'[NetworkDense.fwd()] [{layer.__class__.__name__}] {layer.outputs.shape}'))

        # Update network confidence scores
        self.pred_classes = self.layers[-1].outputs

    def bwd(self, loss_dvals: np.ndarray) -> None:
        
        for layer in self.layers:
            pass




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
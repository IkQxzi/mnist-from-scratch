'''
Created: Sprint 1
Last Modified: Sprint 1
Purpose: Create and propogate neural network
'''

from .base_classes import LayerGeneric

import time
from typing import Dict, List, Union

import numpy as np


class NetworkNeural:

    def __init__(self, n_layers: int, n_layer_neurons: list, n_network_inputs: int) -> None:
        '''
        Initialise a neural network with a number of layers, and number of neurons in each layer.

        Keyword Args:
            n_layers -- The number of layers in the network (input, hidden & output).
            n_layer_neurons -- The number of neurons in each layer respectively (must be same size as n_layers).
            n_network_inputs -- The number of inputs to the network (ie the first layer).
        
        Attributes:
            layers -- A list of the individual layer objects initialised.
        '''

        if n_layers != len(n_layer_neurons):
            raise IndexError('The number of layers must match the length of n_layer_neurons!')

        self.layers: List[object] = self._init_layers(n_layers, n_layer_neurons, n_network_inputs)


    @staticmethod
    def _init_layers(n_layers: int, n_layer_neurons: list, n_network_inputs: int) -> List[object]:
        '''Initialise layers using kword args from __init__().'''

        layers: List[object] = []

        # The first layer must be handled separately because the number of inputs to each layer
        # is defined by the user (with n_network_inputs)
        n_layer1_neurons = n_layer_neurons[0]
        layer = LayerDense(n_layer1_neurons, n_network_inputs)

        # The other layers can be done together because the number of inputs to each layer
        # is simply the number of neurons in the previous layer
        for ind in range(1, n_layers+1): 

            n_neurons = n_layer_neurons[ind]
            n_inputs = n_layer_neurons[ind-1]
            layer = LayerDense(n_neurons, n_inputs)
            layers.append(layer)
        
        return layers
            
# Inherit from LayerGeneric later
class LayerDense():

    def __init__(self, n_neurons: int, n_inputs: int) -> None:
        '''
        Create a dense (fully-connected) layer with random weights and biases.
        
        Keyword Args: 
            n_neurons -- The number of neurons for the layer to have.
            n_inputs -- The number of inputs the layer should expect. 

        Attributes:
            self.weights -- The ndarray of random weights for each input with shape (n_neurons, n_inputs).
            self.biases -- The ndarray of random biases for each neuron with shape (n_neurons, 1).
        '''

        self.weights: np.ndarray = np.random.rand(n_neurons, n_inputs)
        self.biases: np.ndarray = np.random.rand(n_neurons, 1)

    
    def fwd(self, inputs: np.ndarray):
        '''
        Calculate the dot products of the inputs and weights, and add the biases.
        
        Keyword Args:
            inputs -- np ndarray of a complementary shape to the layer weights.
        
        Attributes:
            inputs -- same as kword arg.
            outputs -- np ndarray of inputs ⋅ weights + biases.
        '''

        self.inputs = inputs

        self.outputs = np.dot(inputs, self.weights) + self.biases


layer1 = LayerDense(3, 4)

test_data = np.array([[0.3, 0.56, -0.3, 0.5],
                      [5.3, 2.5, -3.5, 2.7],
                      [3.2, 6.4, 6.2, 0.3]])
'''
Notes: Perhaps move activation functions to other files 
- Move something to other files
'''
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

    def __init__(self, layer_arch: Sequence[int | str], input_size: int, leak_rate: float | None = None) -> None:
        '''
        Initialise a neural network with a number of layers, and number of neurons in each layer.

        Keyword Args:
            layer_arch -- The type of each network layer e.g. (256, ReLU, 16, ReLU, 10, Softmax).
            input_size -- The number of inputs to the network (ie the first layer).
            leak_rate -- The leak rate required for LeakyReLU (only).
        
        Attributes:
            layers -- A list of the individual layer objects initialised.
        '''
        
        # Check if Leaky ReLU is used -- then check for leak_rate
        for arch in layer_arch:
            if type(arch) == str and arch.lower() in ['leakyrelu', 'leaky_relu'] and not leak_rate:
                raise NotImplementedError('LeakyReLU layer requires leak_rate parameter to be specified.')    
        
        # Network architecture attributes
        self.layer_arch = layer_arch
        self.input_size = input_size
        self.leak_rate = leak_rate

        print(debug_style(f'layer arch: {self.layer_arch}'))

        self.layers = self.init_layers() # Both dense & activation layers
        
        self.conf_scores = np.array([]) # Will be updated after forwarding

        # print(debug_style(f'layers: {self.layers}'))


    @staticmethod
    def _init_dense_layers(n_layer_neurons: list[int], input_size: int) -> list[LayerDense]:
        '''Initialise layers using kword args from __init__().'''

        layers: list[LayerDense] = []

        # The first layer must be handled separately because the number of inputs to each layer
        # is defined by the user (with input_size)
        n_layer1_neurons = n_layer_neurons[0]
        layer1 = LayerDense(n_layer1_neurons, input_size)
        layers.append(layer1)

        print(debug_style(f'[NetworkDense._init_dense_layers()] Layer 0: [{n_layer1_neurons}, {input_size}]'))

        # The other layers can be done together because the number of inputs to each layer
        # is simply the number of neurons in the previous layer
        for ind in range(1, len(n_layer_neurons)):

            n_neurons = n_layer_neurons[ind]
            n_inputs = n_layer_neurons[ind-1]

            print(debug_style(f'[NetworkDense._init_dense_layers()] Layer {ind}: [{n_neurons}, {n_inputs}]'))

            layer = LayerDense(n_neurons, n_inputs)
            layers.append(layer)
        
        return layers
    
    @staticmethod
    def _init_activation_layers(act_funcs: list[str], leak_rate: float | None = None) -> list[Activation]:
        '''
        Initialise and return a list of activation objects.

        Keyword Args:
            act_funcs: list of activation functions to use (see functions/activations).
        '''

        act_layers: list[Activation] = []

        # Create array of activation layers
        for act_num, act in enumerate(act_funcs):
            
            match act.lower():

                case 'relu':
                    activ = ReLU()
                    act_layers.append(activ)

                case 'sigmoid':
                    activ = Sigmoid()
                    act_layers.append(activ)

                case 'tanh':
                    activ = Tanh()
                    act_layers.append(activ)

                case 'leaky_relu' | 'leakyrelu' | 'reluleaky' | 'relu_leaky':
                    activ = ReLULeaky(leak_rate) # type: ignore
                    act_layers.append(activ)

                case 'softmax':
                    activ = Softmax()
                    act_layers.append(activ)

                case _:
                    raise NotImplementedError(f"Activation '{act}' doesn't exist.")
            
        return act_layers


    def init_layers(self):
        '''
        Initialise dense and activation layers in self.layers.

        Modified:
            self.layers
        '''
        
        # self.layer_arch = (256, ReLU, 16, ReLU, 10, Softmax)

        # Initialise dense layers
        n_layer_neurons: list[int] = self.layer_arch[0::2] # type: ignore
        dense_layers: list[LayerDense] = self._init_dense_layers(n_layer_neurons=n_layer_neurons, 
                                                                 input_size=self.input_size)
        
        # Initialise activation (incl. loss) layers
        act_funcs: list[str] = self.layer_arch[1::2] # type: ignore
        if self.leak_rate:
            act_layers: list[Activation] = self._init_activation_layers(act_funcs=act_funcs,
                                                                        leak_rate=self.leak_rate)
        else:
            act_layers: list[Activation] = self._init_activation_layers(act_funcs=act_funcs)

        # Combine dense & activation layers into one list
        layers: list[LayerDense | Activation] = []
        for dense_layer, act_layer in zip(dense_layers, act_layers):
            layers.extend([dense_layer, act_layer])

        return layers

    
    def fwd(self, inputs: np.ndarray):
        '''
        Forward a set of inputs through the entire network.

        Keyword Args:
            inputs -- The inputs to the network
        '''

        # The inputs to the network need be passed only into the first layer
        # not any subsequent ones
        print(debug_style(f'[NetworkDense.fwd()] Current Layer Index: 0'))
        layer1 = self.layers[0]
        layer1.fwd(inputs)

        # For each subsequent layer, forward the outputs of the previous layer as its inputs
        for layer_ind, layer in enumerate(self.layers[1:]):

            
            print(debug_style(f'[NetworkDense.fwd()] Current Layer Index: {layer_ind+1}'))

            prev_layer = self.layers[layer_ind] # Uses index of new sliced list
            layer.fwd(prev_layer.outputs)

        # Update network confidence scores
        self.conf_scores = self.layers[-1].outputs



layer1 = LayerDense(3, 4)

test_data = np.array([[0.3, 0.56, -0.3, 0.5],
                      [5.3, 2.5, -3.5, 2.7],
                      [3.2, 6.4, 6.2, 0.3]])

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
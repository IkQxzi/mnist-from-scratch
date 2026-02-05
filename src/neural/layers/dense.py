import time

import cupy as cp
import numpy as np

from neural.core.base import Layer


class LayerDense(Layer):

    def __init__(self, n_neurons: int, n_inputs: int) -> None:
        '''
        Create a dense (fully-connected) layer with random weights and biases.
        
        Keyword Args: 
            n_neurons -- The number of neurons for the layer to have (set to None if params present).
            n_inputs -- The number of inputs the layer should expect (set to None if params present). 

        Attributes:
            self.weights -- The ndarray of (random) weights for each input with shape (n_inputs, n_neurons).
            self.biases -- The ndarray of (random) biases for each neuron with shape (n_neurons, 1).
        '''

        # For metadata
        self.kwargs = {'n_neurons': n_neurons, 'n_inputs': n_inputs}

        # W ~ N(0,1)
        self.weights: cp.ndarray = cp.random.randn(n_neurons, n_inputs, dtype='float32')
        
        # He initialisation: W ~ N(0, 2/n_inputs)
        self.weights *= cp.sqrt(2/n_inputs) 

        # Keep as float32 because will be populated later
        self.biases: cp.ndarray = cp.zeros((1, n_neurons), dtype='float32')

    def load_params(self, weights:cp.ndarray, biases: np.ndarray):
        '''Override random weights & biases with pretrained ones.'''
        self.weights = weights
        self.biases = biases

    
    def fwd(self, inputs: cp.ndarray): # pyright: ignore[reportIncompatibleMethodOverride]
        '''
        Calculate the dot products of the inputs and weights, and add the biases.
        
        Keyword Args:
            inputs -- cp ndarray of the same shape as layer weights (n_neurons, n_inputs).
        
        Attributes:
            inputs -- same as kword arg.
            outputs -- cp ndarray of inputs ⋅ weights + biases.
        '''

        self.inputs = inputs

        self.outputs = cp.dot(inputs, self.weights.T) + self.biases

    
    def bwd(self, chain_dvals: cp.ndarray) -> None:
        '''
        Calculate the partial derivatives of the outputs w.r.t to different parameters.
        
        Keyword Args:
            chain_dvals - derivatives of layer ahead.
        
        Attributes:
            weight_dvals -- derivatives of outputs w.r.t weights.
            input_dvals -- derivatives of outputs w.r.t inputs.
            bias_dvals -- derivatives of outputs w.r.t biases.
        '''

        # derivative of weights = inputs
        local_weight_dvals = self.inputs
        self.weight_dvals = cp.dot(chain_dvals.T, local_weight_dvals)

        # derivative of inputs = weights
        local_input_dvals = self.weights
        self.input_dvals = cp.dot(chain_dvals, local_input_dvals)

        # derivative of biases = 1; calculate batch average
        self.bias_dvals = cp.average(chain_dvals, axis=0)

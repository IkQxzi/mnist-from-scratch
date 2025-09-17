'''
Created: Sprint 1
Last Modified: Sprint 1
Purpose: Create and propogate neural network
'''

import time

import numpy as np


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
            inputs -- same as kword arg
            outputs -- np ndarray of inputs ⋅ weights + biases
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
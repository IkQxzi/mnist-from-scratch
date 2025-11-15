import numpy as np


# Inherit from LayerGeneric later
class LayerDense():

    def __init__(self, n_neurons: int, n_inputs: int) -> None:
        '''
        Create a dense (fully-connected) layer with random weights and biases.
        
        Keyword Args: 
            n_neurons -- The number of neurons for the layer to have.
            n_inputs -- The number of inputs the layer should expect. 

        Attributes:
            self.weights -- The ndarray of random weights for each input with shape (n_inputs, n_neurons).
            self.biases -- The ndarray of random biases for each neuron with shape (n_neurons, 1).
        '''

        self.weights: np.ndarray = np.random.rand(n_neurons, n_inputs)
        
        # Normalise weights to be N(0, 1)
        self.weights -= self.weights.mean(axis=1, keepdims=True)
        
        self.biases: np.ndarray = np.zeros((1, n_neurons))

    
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

        self.outputs = np.dot(inputs, self.weights.T) + self.biases

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


        # For debugging/printing later 
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs

        self.weights: np.ndarray = np.random.rand(n_neurons, n_inputs)
        self.weights -= self.weights.mean(axis=1, keepdims=True) # Normalise weights to be N(0, 1)
        
        self.biases: np.ndarray = np.zeros((1, n_neurons))

    
    def fwd(self, inputs: np.ndarray):
        '''
        Calculate the dot products of the inputs and weights, and add the biases.
        
        Keyword Args:
            inputs -- np ndarray of the same shape as layer weights (n_neurons, n_inputs).
        
        Attributes:
            inputs -- same as kword arg.
            outputs -- np ndarray of inputs ⋅ weights + biases.
        '''

        self.inputs = inputs

        self.outputs = np.dot(inputs, self.weights.T) + self.biases

    
    def bwd(self, chain_dvals: np.ndarray) -> None:
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
        self.weight_dvals = np.dot(local_weight_dvals, chain_dvals.T)

        # derivative of inputs = weights
        local_input_dvals = self.weights
        self.input_dvals = np.dot(local_input_dvals, chain_dvals.T)

        # derivative of biases = 1 
        local_bias_dvals = np.ones_like(self.biases)
        self.bias_dvals = np.dot(local_bias_dvals, chain_dvals.T)



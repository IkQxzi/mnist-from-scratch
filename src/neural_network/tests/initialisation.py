import random
import numpy as np
from ..core import NetworkNeural

class NetworkExample:
    '''
    Attributes:
        n_network_inputs -- As argument for __init__().
        n_layers -- As argument for __init__().
        n_layer_neurons -- A list of length n_layers of random ints.
        test_data -- Random pos/neg floats to be passed in.

    Methods: 
        forward_test()
        gen_rand_network() [staticmethod]
        gen_test_data() [staticmethod]

    '''


    def __init__(self, n_layers: int, n_network_inputs: int):
        '''
        Initialise an example network and test data.

        Keyword Args: 
            Keyword Args: 
            n_layers -- The number of layers to generate random neuron numbers for.
            n_network_inputs -- The number of inputs to the first layer of the network.
            
        '''
        


        # Generate layers
        self.n_network_inputs = n_network_inputs
        self.n_layers = n_layers
        self.n_layer_neurons = self.gen_rand_network(n_layers)

        # Generate test data
        self.test_data = self.gen_test_data(n_network_inputs, n_layer1_neurons=self.n_layer_neurons[0])

    
    def forward_test(self):
        '''
        Test if forward pass works for the number of inputs and layers specified for this instance.

        Attributes Modified:
            network -- NetworkNeural using attributes defined in __init__()

        '''

        self.network = NetworkNeural(self.n_layers, self.n_layer_neurons, self.n_network_inputs)
        self.network.fwd(self.test_data)

        # Only runs if no error has been raised
        print(f"Forward passes worked succesfully for {self.n_layers} layers.\n"
          f"Neuron counts of every layer: {self.n_layer_neurons}")




    @staticmethod
    def gen_rand_network(n_layers: int) -> list[int]:
        '''
        Generate a network of n_layers with random numbers of neurons in each layer.
        
        Keyword Args: 
            n_layers -- The number of layers to generate random neuron numbers for.

        Returns:
            n_layer_neurons -- A list of length n_layers of random ints.
        '''

        n_layer_neurons = []
        for n in range(n_layers):
            n_neurons = random.randint(1, 10)
            n_layer_neurons.append(n_neurons)
        
        return n_layer_neurons


    @staticmethod
    def gen_test_data(n_network_inputs: int, n_layer1_neurons: int) -> np.ndarray:
        '''
        Generate random test data of size (n_network_inputs, n_layer1_neurons) to be passed into the network.

        Keyword Args: 
            n_network_inputs -- The number of datapoints for each neuron.
            n_layer1_neurons -- The number of neurons in the first layer of the network.

        Returns:
            test_data -- Randomly generated test_data, a mix of pos/neg floats.
        '''

        return np.random.rand(n_network_inputs, n_layer1_neurons)
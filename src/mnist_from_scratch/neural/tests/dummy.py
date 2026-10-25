'''
Generate dummy networks for testing purposes.

Created: Sprint 1?
Modified: Sprint 2
'''

import random
from typing import Sequence

import numpy as np

from mnist_from_scratch.neural.layers import activations
from mnist_from_scratch.neural.core.network import NetworkDense
from mnist_from_scratch.tools.debugging import debug_style
# from neural.core import NetworkNeural


def gen_batch_data(batch_size: int, layer1_size: int) -> np.ndarray:
    '''
    Generate random test data of shape (layer1_size, batch_size) to be passed into the network.

    Keyword Args: 
        batch_size -- The number of datapoints for each neuron.
        layer1_size -- The number of neurons in the first layer of the network.

    Returns:
        test_data -- Randomly generated test_data, a mix of pos/neg floats.
    '''

    # input size = no. of batches
    return np.random.rand(layer1_size, batch_size)


# should be true classes not pred
def gen_test_one_hot(layer_arch: list, batch_size: int):
    '''
    Generate random o

    Keyword Args: 
        layer_arch -- The architecture of the network.
        batch_size -- The number of samples in each batch.

    Returns:
        one_hot_arr -- Randomly generated one-hot encoded vectors.
    '''

    # Number of neurons in output layer 
    neuron_num = layer_arch[-2]

    # Random 1D arr of indices from 0 to no. of neurons
    rand_indices = np.random.randint(low=0, high=neuron_num, size=(batch_size))

    # Create complementary arr of one-hot values to output layer 
    one_hot_arr = np.zeros(shape=(batch_size, neuron_num))
    one_hot_arr[[x for x in range(batch_size)], rand_indices] = 1

    return one_hot_arr

class NetworkDenseExample:

    @staticmethod
    def gen_layer_arch(layer_num: int, neuron_upper_b: int) -> Sequence[int | str]:
        '''
        Generate random NetworkDense params (ending with Softmax) for given no. of layers & upper bound of neuron count.

        Keyword Args:
            layer_num -- The number of layers (not including activation or input) in the network.
            neuron_upper_b -- The upper bound for the (random) number of neurons which can be generated for a layer.
        '''

        layer_arch: Sequence[int | str] = []

        # Only want Softmax on final output, so remove it from list of activations
        acts = activations.__all__
        if 'Softmax' in acts: # Allows for repeated testing without ValueError
            acts.remove('Softmax')

        for _ in range(layer_num):

            # Random neuron number
            layer_arch.append(random.randint(1, neuron_upper_b))

            # Generate a random activation (name) from the list
            act_num = random.randint(0, len(acts)-1)
            layer_arch.append(acts[act_num])
        
        # Replace final activation with Softmax
        layer_arch.pop(-1)
        layer_arch.append('Softmax')

        return layer_arch


    def __init__(self, batch_size: int, layer_num: int, neuron_upper_b, layer_arch: list | None = None) -> None:
        '''
        Initialise a random NetworkDense object to be used for testing.

        Keyword Args:
            batch_size -- The number of inputs to the network (the first layer).
            layer_num -- The number of layers (not including activation or input) in the network.
            neuron_upper_b -- The upper bound for the (random) number of neurons which can be generated for a layer.
            layer_arch -- The optional shape of the network

        Attributes:
            network -- The NetworkDense object.
        '''

        self.batch_size = batch_size
        
        # Use layer architecture if specified, otherwise randomise
        if layer_arch:
            self.layer_arch = layer_arch
        else:
            self.layer_arch = self.gen_layer_arch(layer_num, neuron_upper_b)
        
        self.network_inputs = gen_batch_data(batch_size=self.batch_size, layer1_size=self.layer_arch[0]) # pyright: ignore[reportArgumentType]

        # Leaky ReLU requires a leak rate (0 < float < 1)
        if 'ReLULeaky' in self.layer_arch:
            self.network = NetworkDense(self.layer_arch, self.batch_size, leak_rate=random.random())
        else:
            self.network = NetworkDense(self.layer_arch, self.batch_size)


# class NetworkExample:
#     '''
#     Attributes:
#         batch_size -- As argument for __init__().
#         n_layers -- As argument for __init__().
#         n_layer_neurons -- A list of length n_layers of random ints.
#         test_data -- Random pos/neg floats to be passed in.

#     Methods: 
#         forward_test()
#         gen_rand_network() [staticmethod]
#         gen_batch_data() [staticmethod]

#     '''

#     def __init__(self, n_layers: int, batch_size: int):
#         '''
#         Initialise an example network and test data.

#         Keyword Args: 
#             Keyword Args: 
#             n_layers -- The number of layers to generate random neuron numbers for.
#             batch_size -- The number of inputs to the first layer of the network.
            
#         '''        

#         # Generate layers
#         self.batch_size = batch_size
#         self.n_layers = n_layers
#         self.n_layer_neurons = self.gen_rand_network(n_layers)

#         # Initialise network
#         self.network = NetworkNeural(self.n_layers, self.n_layer_neurons, self.batch_size)

#         # Generate test data
#         layer1_size = self.n_layer_neurons[0]
#         self.test_data = self.gen_batch_data(batch_size, layer1_size)

    
#     def forward_test(self):
#         '''
#         Test if forward pass works for the number of inputs and layers specified for this instance.

#         Attributes Modified:
#             network -- Forwarded data to each layer.

#         '''
        
#         self.network.fwd(self.test_data)

#         # Only runs if no error has been raised
#         print(f"Forward passes worked succesfully for {self.n_layers} layers with {self.batch_size} inputs.\n"
#           f"Neuron counts of every layer: {self.n_layer_neurons}.\n")


#     @staticmethod
#     def gen_rand_network(n_layers: int) -> list[int]:
#         '''
#         Generate a network of n_layers with random numbers of neurons in each layer.
        
#         Keyword Args: 
#             n_layers -- The number of layers to generate random neuron numbers for.

#         Returns:
#             n_layer_neurons -- A list of length n_layers of random ints.
#         '''

#         n_layer_neurons = []
#         for n in range(n_layers):
#             n_neurons = random.randint(1, 10)
#             n_layer_neurons.append(n_neurons)
        
#         return n_layer_neurons

__all__ = ["NetworkDenseExample", "gen_batch_data"]
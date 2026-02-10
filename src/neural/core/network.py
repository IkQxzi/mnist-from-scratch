
'''
Created: Sprint 1
Last Modified: Sprint 1
Purpose: Create and propogate neural network
'''

# Built-in imports
import json

# External imports
import cupy as cp

# Local imports
# from .base_classes import LayerGeneric
from neural.core.base import Network
from neural.layers import LayerDense
from neural.layers.activations import * # Probably bad practice

from tools.debugging import debug_style, warning_style


class NetworkDense(Network):

    def __init__(self, layers: list) -> None:
        '''
        Initialise a neural network using the specified architecture.

        Keyword Args:
            layers -- The architecture of the network (as objects).
            param_file -- A file which jcontains the parameters of the network.
        
        Attributes:
            layers -- A list of the layer objects.
            layer_arch -- A clean str list of the layer names.
            probs -- A list of the networks predictions.
            conf_scores -- The confidence of each prediction of the network.
        '''

        self.layers = layers # Both dense & activation layers

        # List of layer names (don't know how necessary)
        # TODO: make like above using list slicing
        self.layer_arch: list[str] = []
        # for i, arch in enumerate(sjelf.layers):
        #     if i % 2 == 0: # Indexes of dense layers [0, 2, 4]
        #         self.layer_arch.append(f'LayerDense{arch.n_neurons, arch.n_inputs}')
        #     else:
        #         self.layer_arch.append(f'{arch.__class__.__name__}()')

        # print(debug_style(f'Layer arch: {self.layer_arch}'))

        # Will be updated after forwarding
        self.probs = cp.array([], dtype=float) 
        self.conf_scores = cp.array([], dtype=float)

    @classmethod
    def load_pretrained(cls, param_path: str):
        '''
        Load a NetworkDense from a file of model params.
        
        :param param_path: Path to .npz parameter file.
        :type param_path: str
        '''

        layers = []

        container = cp.load(param_path)

        metadata = json.loads(container['metadata'])
        layers = container['layers']

        for layer in layers:
            
            layer_class = getattr(layer, layer['type'])
            layer_kwargs = layer['kwargs']

            if layer_class is LayerDense:

                # Get the keys for weights, biases
                keys = layer['keys']
                w_key = keys['w_key']
                b_key = keys['b_key']

                # create dense layer and load pretrained params (from container)
                dense = LayerDense(**layer_kwargs)
                dense.load_params(
                                weights=container[w_key],
                                biases=container[b_key]
                                  )
        
                layers.append(dense)
            
            # Other funcs with params that aren't dense
            elif layer_kwargs:
                layers.append(layer_class(**layer_kwargs))

            # TODO: find another more elegant solution instead of if statement
            # See if there is a way to pass **kwargs and have it be ignored
            else:
                layers.append(layer_class())

        return cls(layers)
            

    def _calc_conf_scores(self) -> None:
        '''Calculate and update self.conf_scores using self.probs.'''

        output_sum = cp.sum(self.probs, axis=1) # Sum of each sample
        self.conf_scores = cp.ones_like(self.probs)
        for i in range(output_sum.shape[0]):
            self.conf_scores[i] = (self.probs[i] / output_sum[i])

        self.conf_scores *= 100 # dec to %
        self.conf_scores[self.conf_scores < 0.01] = 0 # Replace small values with 0


    def fwd(self, inputs: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        '''
        Forward a set of inputs through the entire network.

        Keyword Args:
            inputs -- The inputs to the network.

        Returns:
            probs -- The predictions of the network.
            conf_scores -- How confident the network is for each prediction.
        '''

        # The inputs to the network need be passed only into the first layer
        # not any subsequent ones
        # print(debug_style(f'[NetworkDense.fwd()] Current Layer Index: 0'))
        layer1 = self.layers[0]
        layer1.fwd(inputs)
        # print(debug_style(f'[NetworkDense.fwd()] [{layer1.__class__.__name__}] {layer1.outputs.shape}'))

        # For each subsequent layer, forward the outputs of the previous layer as its inputs
        for layer_ind, layer in enumerate(self.layers[1:]):

            
            # print(debug_style(f'[NetworkDense.fwd()] Current Layer Index: {layer_ind+1}'))

            fwd_layer = self.layers[layer_ind] # Uses index of new sliced list
            layer.fwd(fwd_layer.outputs)

            # print(debug_style(f'[NetworkDense.fwd()] [{layer.__class__.__name__}] {layer.outputs.shape}'))

        # Update network outputs
        self.probs = self.layers[-1].outputs

        self._calc_conf_scores()

        return self.probs, self.conf_scores


    # Rewrite this rubbish
    def bwd(self, **kwargs) -> None:
        '''
        Backpropgate through layers, from output to first layer.
        '''
        
        # Do using kwargs to handle any output activ
        # NOTE: using kwargs is evil
        output_activ = self.layers[-1]
        output_activ.bwd(**kwargs)

        # Number of non-Softmax layers
        n_reg_layers = len(self.layers[:-1])

        # Iterate through layers backwards, from final LayerDense to first LayerDense
        for i, layer in enumerate(self.layers[-2::-1]):
            
            # print(debug_style(f'Backprop: {fwd_layer.__class__.__name__} -> {layer.__class__.__name__}'))
            
            fwd_layer = self.layers[n_reg_layers-i]
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
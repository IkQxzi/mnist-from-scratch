"""Created: Sprint 1
Last Modified: Sprint 1
Purpose: Create and propogate neural network
"""

import cupy as cp

from mnist_from_scratch.neural.core.base import Network
from mnist_from_scratch.neural.layers import activations
from mnist_from_scratch.neural.layers.dense import LayerDense


class NetworkDense(Network):
    def __init__(self, layers: list) -> None:
        """Initialise a neural network using the specified architecture.

        Keyword Args:
            layers -- The architecture of the network (as objects).

        Attributes:
            layers -- A list of the layer objects.
            probs -- A list of the networks predictions.
            conf_scores -- The confidence of each prediction of the network.
        """
        self.layers = layers  # Both dense & activation layers

        # Will be updated after forwarding
        self.probs = cp.array([], dtype=float)
        self.conf_scores = cp.array([], dtype=float)

    @classmethod
    def load_pretrained(cls, raw_net_cont, layer_metadata: dict):
        """Load a NetworkDense using a container from a .npz file.

        :param container: The container from the .npz file after being decoded from json.
        :type container: dict
        """
        # List of layer objects (as in __init__())
        layers = []

        for layer in layer_metadata:
            layer_cls_name = layer["cls_name"]
            layer_kwargs = layer["kwargs"]

            # Create dense layer and load pretrained params (from container)
            if layer_cls_name == "LayerDense":
                # Get the keys for weights, biases
                keys = layer["keys"]
                w_key = keys["w_key"]
                b_key = keys["b_key"]

                # load_params overwrites random params
                dense = LayerDense(**layer_kwargs)

                # Convert weights, biases to cupy arrays before writing
                dense.load_params(
                    weights=cp.asarray(raw_net_cont[w_key]),
                    biases=cp.asarray(raw_net_cont[b_key]),
                )

                layers.append(dense)

            # Other funcs with params that aren't dense
            elif layer_kwargs:
                # Get the activation function from the list of activations
                act_func = getattr(activations, layer_cls_name)
                layers.append(act_func(**layer_kwargs))  # Pass in kwargs

            # TODO: find another more elegant solution instead of if statement
            # See if there is a way to pass **kwargs and have it be ignored
            else:
                # Same as above
                act_func = getattr(activations, layer_cls_name)
                layers.append(act_func())  # No kwargs to pass

        return cls(layers)

    def _calc_conf_scores(self) -> None:
        """Calculate and update self.conf_scores and self.pred_digit using self.probs."""

        # Should be much faster
        self.conf_scores = 100 * self.probs / cp.sum(self.probs, keepdims=True)

    def get_pred_digit_and_conf(self) -> tuple[int, float]:
        """Get the network's main prediction and confidence score (unrounded)."""
        # Explicitly cast to int (returns arr by default)
        pred_digit = int(cp.argmax(self.conf_scores))

        # cp.max always returns an arr, so cast to float
        main_conf_score = float(cp.max(self.conf_scores))

        return pred_digit, main_conf_score

    def fwd(self, inputs: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        """Forward a set of inputs through the entire network.

        Keyword Args:
            inputs -- The inputs to the network.

        Returns:
            probs -- The predictions of the network.
            conf_scores -- How confident the network is for each prediction.
        """
        # The inputs to the network need be passed only into the first layer
        # not any subsequent ones
        # print(debug_style(f'[NetworkDense.fwd()] Current Layer Index: 0'))
        layer1 = self.layers[0]
        layer1.fwd(inputs)
        # print(debug_style(f'[NetworkDense.fwd()] [{layer1.__class__.__name__}] {layer1.outputs.shape}'))

        # For each subsequent layer, forward the outputs of the previous layer as its inputs
        for layer_ind, layer in enumerate(self.layers[1:]):
            # print(debug_style(f'[NetworkDense.fwd()] Current Layer Index: {layer_ind+1}'))

            fwd_layer = self.layers[layer_ind]  # Uses index of new sliced list
            layer.fwd(fwd_layer.outputs)

            # print(debug_style(f'[NetworkDense.fwd()] [{layer.__class__.__name__}] {layer.outputs.shape}'))

        # Update network outputs
        self.probs = self.layers[-1].outputs

        self._calc_conf_scores()

        return self.probs, self.conf_scores

    # Rewrite this rubbish
    def bwd(self, **kwargs) -> None:
        """Backpropgate through layers, from output to first layer."""
        # Do using kwargs to handle any output activ
        # NOTE: using kwargs is evil
        output_activ = self.layers[-1]
        output_activ.bwd(**kwargs)

        # Number of non-Softmax layers
        n_reg_layers = len(self.layers[:-1])

        # Iterate through layers backwards, from final LayerDense to first LayerDense
        for i, layer in enumerate(self.layers[-2::-1]):
            # print(debug_style(f'Backprop: {fwd_layer.__class__.__name__} -> {layer.__class__.__name__}'))

            fwd_layer = self.layers[n_reg_layers - i]
            layer.bwd(fwd_layer.input_dvals)


# layer1 = LayerDense(3, 4)

# test_data = np.array([[0.3, 0.56, -0.3, 0.5],
#                       [5.3, 2.5, -3.5, 2.7],
#                       [3.2, 6.4, 6.2, 0.3]])

"""
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


"""

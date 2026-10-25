"""
Stores the tests for the neural network.

Created: Sprint 2
Last Modified: Sprint 2
"""

import random
from typing import Sequence

import numpy as np

from mnist_from_scratch.neural.core.network import NetworkDense

# from neural.tests import NetworkExample
from mnist_from_scratch.neural.layers import activations
from mnist_from_scratch.neural.layers.activations import *
from mnist_from_scratch.tools.debugging import debug_style


# Deprecated
def scrap_test() -> None:
    n_layers = 2
    n_network_inputs = 4

    test_data = np.array([[0.5, 3, 25], [4, 5.3, -3], [4.2, -5, 0.7], [4.3, 0.8, 0.3]])
    print(test_data.shape)

    net = NetworkDense(n_layers, [3, 4], n_network_inputs)
    net.fwd(test_data)


# Deprecated
def example_forward_test() -> None:
    # Example input data
    n_layers = 20
    n_network_inputs = 4

    net = NetworkExample(n_layers, n_network_inputs)
    print(debug_style(f"[NetworkExample] Layers: {net.n_layers}"))
    print(debug_style(f"[NetworkNeural] Layers: {len(net.network.layers)}"))
    net.forward_test()


def gen_dense_params(layer_num: int, neuron_upper_bound: int) -> Sequence[int | str]:
    """
    Generate random NetworkDense params for a given number of layers and upper bound of neuron count.

    Keyword Args:
        layer_num -- The number of layers (not including activation or input) in the network.
        neuron_upper_bound -- The upper bound for the (random) number of neurons which can be generated for a layer.
    """

    input_size = random.randint(1, 256)

    layer_arch: Sequence[int | str] = []
    for n in range(layer_num):
        # Random neuron number
        layer_arch.append(random.randint(1, neuron_upper_bound))

        # Generate a random activation (name) from the list of possible activations
        acts = activations.__all__
        num_acts = len(acts)
        act_num = random.randint(0, num_acts - 1)
        act_name = acts[act_num]
        layer_arch.append(act_name)

    return layer_arch

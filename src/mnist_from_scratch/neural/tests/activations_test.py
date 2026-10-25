import numpy as np

from mnist_from_scratch.neural.layers.activations import ReLU, ReLULeaky
from mnist_from_scratch.tools.debugging import debug_style


def leaky_relu_test():

    print(debug_style("Leaky ReLU Test"))
    activ_func = ReLULeaky(leak_rate=0.1)

    # Mix of pos/neg floats
    test_data = np.random.randn(4, 3)
    print(f"test data: {test_data}\n")

    # (Should be) floats between 0 & 1
    activ_func.fwd(test_data)
    print(f"outputs: {activ_func.outputs}\n")


def relu_test():
    print(debug_style("ReLU Test"))
    activ_func = ReLU()

    # Mix of pos/neg floats
    test_data = np.random.randn(4, 3)
    print(f"test data: {test_data}\n")

    # (Should be) floats between 0 & 1
    activ_func.fwd(test_data)
    print(f"outputs: {activ_func.outputs}\n")

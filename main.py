'''
The master file from which other functions and classes are called & instantiated. 
Created: Sprint 1
Last Modified: Sprint 2
'''

import numpy as np
from keras.datasets import mnist

from neural import NetworkDense
from neural.tests import NetworkDenseExample, gen_batch_data, gen_test_one_hot
from neural import get_mnist_data
from neural.tests.data import gen_test_data
from tools.debugging import debug_style

from neural.layers import LayerDense
from neural.layers.activations import * 
from neural.loss import CategoricalCrossEntropy
from neural.optimisers import StochasticGradientDescent


def test_pass() -> None:
    
    batch_size: int = 256
    layer_num = 3
    neuron_upper_b = 16
    loops = 1

    for i in range(loops):
        print(f"\n{debug_style(f'Loop {i+1}')}")
        
        test_net = NetworkDenseExample(batch_size, layer_num, neuron_upper_b)
        # print(debug_style(f'Network inputs:\n {test_net.network_inputs}'))
        
        test_net.network.fwd(test_net.network_inputs)

        print(debug_style(f'[INPUTS]:\n {test_net.network_inputs}'))

        # Print outputs for each layer
        for layer_n, layer in enumerate(test_net.network.layers):
            print(debug_style(f'[{layer_n}][{layer.__class__.__name__}] outputs:\n {layer.outputs}'))

        # Print final outputs
        print(debug_style(f'[FINAL OUTPUTS]:\n{test_net.network.pred_classes}'))
        print(debug_style(f'[Sum across neurons][axis 1]:\n{np.sum(test_net.network.pred_classes, axis=1)}'))

        true_classes = gen_test_one_hot(layer_arch=test_net.layer_arch, batch_size=batch_size) # pyright: ignore[reportArgumentType]

        loss = CategoricalCrossEntropy()
        loss.calculate(test_net.network.pred_classes, true_classes)
        print(f"\n{debug_style(f'[Loss]: {loss.outputs}')}\n")


def manual_bwd(network: NetworkDense, loss: CategoricalCrossEntropy, cor_outputs: np.ndarray) -> None:
    
    smax: Softmax = network.layers[-1] # type: ignore
    smax.bwd(loss.input_dvals, true_classes=cor_outputs)
    print(debug_style(f'Softmax local_dvals:\n{smax.local_input_dvals}\n'
                      f'Softmax dvals (chain rule):\n{smax.input_dvals}'))
    
    final_dense_layer: LayerDense = network.layers[-2] # type: ignore
    final_dense_layer.bwd(smax.input_dvals)
    print(debug_style(f'Dense outputs [Neuron 1]:\n{final_dense_layer.outputs[:, 0]}\n'
                      f'Weight dvals [Neuron 1]:\n{final_dense_layer.weight_dvals[0]}\n'
                      f'Input dvals:\n{final_dense_layer.input_dvals}\n'
                      f'Bias dvals:\n{final_dense_layer.bias_dvals}'))
    
    final_relu: ReLU = network.layers[-3] # type: ignore
    final_relu.bwd(final_dense_layer.input_dvals)
    print(debug_style(f'ReLU dvals ({final_relu.input_dvals.shape}):\n{final_relu.input_dvals}'))


def main() -> None:

    # For readability
    np.set_printoptions(suppress=True, precision=2)

    n_inputs = 784
    n_neurons = 64
    n_batches = 2
    learning_rate = 0.2
    epochs = 1000

    layers = [
        LayerDense(n_neurons, n_inputs),
        ReLU(),
        LayerDense(16, n_inputs=n_neurons),
        ReLU(),
        LayerDense(10, 16),
        Softmax()
    ]

    network = NetworkDense(layers=layers)

    cor_outputs = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])    

    test_data = gen_test_data(n_batches=n_batches, n_inputs=n_inputs)
    
    for _ in range(epochs):

        network.fwd(test_data)

        print(debug_style(f'Network pred_classes {network.pred_classes.shape}:\n'
                        f'{network.pred_classes}\n'
                        f'Confidence scores\n{network.conf_scores}'))

        loss = CategoricalCrossEntropy()
        loss.calculate(network.pred_classes, cor_outputs)
        print(f"\n{debug_style(f'[Loss]: {loss.outputs}')}")

        loss.bwd()
        print(debug_style(f'[Loss Derivative] {loss.input_dvals}'))

        network.bwd(loss_dvals=loss.input_dvals, true_classes=cor_outputs)

        sgd = StochasticGradientDescent(learning_rate=learning_rate, network=network)
        sgd.update_params()


if __name__ == '__main__':
    main()

# main.py




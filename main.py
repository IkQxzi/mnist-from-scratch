'''
The master file from which other functions and classes are called & instantiated. 
Created: Sprint 1
Last Modified: Sprint 2
'''

import numpy as np
from keras.datasets import mnist

from neural import NetworkDense
from neural.tests import NetworkDenseExample, gen_batch_data, gen_test_one_hot
from neural import get_mnist_batches
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
        print(debug_style(f'[FINAL OUTPUTS]:\n{test_net.network.probs}'))
        print(debug_style(f'[Sum across neurons][axis 1]:\n{np.sum(test_net.network.probs, axis=1)}'))

        true_classes = gen_test_one_hot(layer_arch=test_net.layer_arch, batch_size=batch_size) # pyright: ignore[reportArgumentType]

        loss = CategoricalCrossEntropy()
        loss.calculate(test_net.network.probs, true_classes)
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


def toy_test() -> None:

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

        print(debug_style(f'Network probs {network.probs.shape}:\n'
                        f'{network.probs}\n'
                        f'Confidence scores\n{network.conf_scores}'))

        loss = CategoricalCrossEntropy()
        loss.calculate(network.probs, cor_outputs)
        print(f"\n{debug_style(f'[Loss]: {loss.outputs}')}")

        loss.bwd()
        print(debug_style(f'[Loss Derivative] {loss.input_dvals}'))

        network.bwd(loss_dvals=loss.input_dvals, true_classes=cor_outputs)

        sgd = StochasticGradientDescent(learning_rate=learning_rate, network=network)
        sgd.update_params()


def batch_train() -> float:
    
    # For readability
    # np.set_printoptions(suppress=True, precision=3)

    n_inputs = 784
    n_neurons = 64
    batch_size = 32
    learning_rate = 0.001
    epochs = 1875

    layers = [
        LayerDense(n_neurons, n_inputs),
        ReLU(),
        LayerDense(16, n_inputs=n_neurons),
        ReLU(),
        LayerDense(10, 16),
        Softmax()
    ]

    network = NetworkDense(layers=layers)
    # optimiser = StochasticGradientDescent(learning_rate=0.01, network=network)

    (x_train_batches, y_train_batches), (x_test, y_test) = get_mnist_batches(batch_size=batch_size)

    loss = CategoricalCrossEntropy()

    print('Training on 60,000 MNIST... [batch size: 32]')
    for epoch in range(epochs):
        i = epoch

        net_inputs = x_train_batches[i]
        true_classes = y_train_batches[i]

        network.fwd(net_inputs)

        # print(debug_style(f'x_train_batches[i] shape: {net_inputs.shape}\n'
        #                   f'y_train_batches[i] shape: {true_classes.shape}'))
        
        # print(debug_style(f'x_train_batches: {x_train_batches[i, 0, :28]}\n'
        #                   f'y_train_batches: {y_train_batches[i, 0]}'))
        
        # print(debug_style(f'probs:\n {network.probs}'))

        loss.calculate(probs=network.probs, true_classes=true_classes)
        # print(debug_style(f'Loss:\n {loss.outputs}'))
        
        loss.bwd()
        
        network.bwd(loss_dvals=loss.input_dvals, true_classes=true_classes)

        # print(debug_style(f'final layer (Softmax) dvals:\n {network.layers[-1].input_dvals[0]}\n'))
        
        # input('...')
        
        optimiser = StochasticGradientDescent(learning_rate=learning_rate, network=network)
        optimiser.update_params()

        # optimiser.update_params()

    # test pass
    loss_total = 0
    pred_total = 0
    for i in range(x_test.shape[0]):
        
        network.fwd(x_test[i])
        loss.calculate(probs=network.probs, true_classes=y_test[i])
        
        # For monitoring later
        loss_total += np.sum(loss.outputs)
        
        # Add 1 when predictin is correct
        pred_ind = np.argmax(network.probs)
        if y_test[i][pred_ind] == 1:
            pred_total += 1
    
    avg_accuracy = 100 * (pred_total / 10_000)
    print(f'Average loss on test set: {loss_total/10000}')
    print(f'Average accuracy on test set: {avg_accuracy}')

    return avg_accuracy


def main() -> None:
    
    runs = 0
    acc_current = 0
    acc_total = 0
    fails = 0

    while True:
        try:
            runs += 1
            acc_current = batch_train()
            acc_total += acc_current
            
            if acc_current < 10:
                fails +=1
                acc_total -= acc_current
                runs -= 1

        except KeyboardInterrupt:
            print(f'Number of runs: {runs}\nAverage accuracy: {acc_total / runs}\nNumber of fails: {fails}')
            exit()


if __name__ == '__main__':
    main()
            

# main.py




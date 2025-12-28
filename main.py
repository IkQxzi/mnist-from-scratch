'''
The master file from which other functions and classes are called & instantiated. 
Created: Sprint 1
Last Modified: Sprint 2
'''

import time
import csv
from typing import Iterator

# from jax import jit
import numpy as np
from keras.datasets import mnist

from neural import NetworkDense
from neural import get_mnist_batches, get_mnist_data
from tools.debugging import debug_style, warning_style
from neural.analysis import get_best_net_acc
from neural.core import save_params
from neural.layers import LayerDense
from neural.layers.activations import * 
from neural.loss import CategoricalCrossEntropy
from neural.optimisers import StochasticGradientDescent


def epoch_train_test(learning_rate: float, epochs: int, w_decay: float) -> tuple[NetworkDense, float]:

    # For readability
    np.set_printoptions(suppress=True, precision=3)

    n_inputs = 784
    net_neurons = [784, 128, 10]
    batch_size = 32
    learning_rate = learning_rate
    w_decay = w_decay # Halfed because not using traditional L2
    epochs = epochs

    layers = [
        LayerDense(net_neurons[0], n_inputs),
        ReLU(),
        LayerDense(net_neurons[1], n_inputs=net_neurons[0]),
        ReLU(),
        LayerDense(net_neurons[2], net_neurons[1]),
        Softmax()
    ]

    net = NetworkDense(layers)
    loss = CategoricalCrossEntropy()
    optimiser = StochasticGradientDescent(net, learning_rate, w_decay)

    (x_train_batches, y_train_batches), (x_test, y_test) = get_mnist_batches(batch_size)

    print(f'Training on 60,000 MNIST, {epochs} epochs...')

    # 1 epoch is the entire training set being forwarded
    for epoch in range(epochs):
        print(f"[{time.strftime('%H:%M:%S')}] Epoch: {epoch+1}")

        for x_train, y_train in zip(x_train_batches, y_train_batches):
            
            # np.random.shuffle(x_train)
            # np.random.shuffle(y_train)

            # Fwd pass
            net.fwd(x_train)
            loss.calculate(probs=net.probs, true_classes=y_train)

            # Bwd pass
            loss.bwd()
            net.bwd(loss_dvals=loss.input_dvals, true_classes=y_train)

            # Optimser update
            optimiser.update_params()
        

    avg_acc = net_test(net, x_test, y_test)

    return net, avg_acc


def net_test(network: NetworkDense, x_test: np.ndarray, y_test: np.ndarray) -> float:

    loss = CategoricalCrossEntropy()

    # test pass
    loss_total = 0
    pred_total = 0
    for i in range(x_test.shape[0]):
        
        network.fwd(x_test[i])
        loss.calculate(probs=network.probs, true_classes=y_test[i])
        
        # For monitoring later
        loss_total += np.sum(loss.outputs)
        
        # Add 1 when prediction is correct
        pred_ind = np.argmax(network.probs)
        if y_test[i][pred_ind] == 1:
            pred_total += 1
    
    avg_accuracy = 100 * (pred_total / x_test.shape[0])
    print(f'Average loss on test set: {loss_total/x_test.shape[0]}')
    print(f'Average accuracy on test set: {avg_accuracy}')

    return avg_accuracy


def main() -> None:
    
    l_rates = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    epochs = 30
    w_decay = 0

    counter = -1
    best_acc = get_best_net_acc(path='logs//best_networks.csv')

    while True:
        counter +=1
        i = counter % 6

        l_rate = l_rates[i]
        net, cur_acc = epoch_train_test(l_rate, epochs=epochs, w_decay=w_decay)
        
        # Write accuracy and time in file:
        with open('logs//best_networks.txt', 'a') as file:

            time_now = time.strftime('%H:%M:%S')

            file.write(f'[{time_now}] learning rate: {l_rate}, w_decay: {w_decay}, accuracy: {cur_acc:.2f}, best: {cur_acc > best_acc}\n')

        # We want highest accuracy network only
        if cur_acc > best_acc:
            best_acc = cur_acc

            save_params(network=net, path=f'logs//layer_params_{cur_acc:.2f}_{l_rate}')
            

    
def test_saving() -> None:
    net_params = np.load('logs//layer_params.npz')

    param_names = sorted(net_params.files)
    print(param_names)
    
    for param_set in param_names:
        print(net_params[param_set].shape)


if __name__ == '__main__':
    main()
            

# main.py




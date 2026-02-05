'''
The main file from which other functions and classes are called & instantiated. 
Created: Sprint 1
Last Modified: Sprint 7
'''

# Standard modules
import time

# External libraries
import numpy as np
import cupy as cp
from keras.datasets import mnist

# Internal imports
from neural import NetworkDense, LayerDense, ReLU, Softmax, CCE, SGD, save_params
from neural.core import get_mnist_batches
from training import get_best_net_acc, log_training_csv
from tools.debugging import debug_style, warning_style



def epoch_train_test(learning_rate: float, epochs: int, w_decay: float, batch_size: int) -> tuple[NetworkDense, float]:

    # For readability
    np.set_printoptions(suppress=True, precision=3)

    n_inputs = 784
    net_neurons = [784, 128, 10]
    batch_size = batch_size
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
    loss = CCE()
    optimiser = SGD(net, learning_rate, w_decay)

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
            net.bwd(chain_dvals =loss.input_dvals, true_classes=y_train)

            # Optimser update
            optimiser.update_params()
    
        assert net.layers[0].weights.dtype == 'float32', f'Not float32: {net.layers[0].weights.dtype}'
        assert net.layers[0].biases.dtype == 'float32', f'Not float32: {net.layers[0].biases.dtype}'
        assert net.layers[0].weight_dvals.dtype == 'float32', f'Not float32: {net.layers[0].weight_dvals.dtype}'
        assert net.layers[0].bias_dvals.dtype == 'float32', f'Not float32: {net.layers[0].bias_dvals.dtype}'
        

    avg_acc = net_test(net, x_test, y_test)


    return net, avg_acc


def net_test(network: NetworkDense, x_test: np.ndarray, y_test: np.ndarray) -> float:

    loss = CCE()

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

    # Promising rates to test
    l_rates = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    epochs = 1
    w_decay = 0
    batch_size = 32
    logpath = 'logs//gpu_best_networks.csv'
    logging = False

    counter = -1
    best_acc = get_best_net_acc(path=logpath)

    while True:
        best = False
        counter +=1
        i = counter % len(l_rates) # To make it cycle learning rates

        l_rate = l_rates[i]

        start_time = time.time()
        net, cur_acc = epoch_train_test(l_rate, epochs=epochs, w_decay=w_decay, batch_size=batch_size)
        time_elapsed = time.time() - start_time

        # We want highest accuracy network only
        best_acc = 0
        if cur_acc > best_acc:
            best = True
            best_acc = cur_acc

            save_params(network=net, path=f'logs//layer_params_{cur_acc:.2f}_{l_rate}')

        # Write training results to file
        if logging:
            log_training_csv(
                path=logpath, time_taken=time_elapsed, batch_size=batch_size, epochs=epochs, l_rate=l_rate, 
                w_decay=w_decay, acc=cur_acc, best=best
                            )
            
        exit()
            


if __name__ == '__main__':
    main()
            

# main.py




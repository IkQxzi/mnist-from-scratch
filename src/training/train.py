'''

'''

import cupy as cp
import numpy as np

from neural.core.network import NetworkDense
from neural.loss.cce import CCE
from neural.optimisers.sgd import SGD

from neural.core.datasets import get_mnist_batches


def train_network(network: NetworkDense, training_data: tuple[cp.ndarray, cp.ndarray], hyperparams: dict, save_net: bool):

    # epochs, batch_size, l_rate, w_decay

    loss = CCE()
    optimiser = SGD(network, hyperparams['l_rate'], hyperparams['w_decay'])
    
    (x_train, y_train) = training_data

    for _ in range(hyperparams['epochs']):
        
        for i in x_train.shape[0]:
            x_batch = x_train[i]
            y_batch = y_train[i]

            # fwd pass
            network.fwd(x_batch)
            loss.calculate(probs=network.probs, true_classes=y_batch)
            loss_vals = loss.outputs

            # bwd pass
            loss.bwd()
            network.bwd(chain_dvals=loss.input_dvals, true_classes=y_train)

            # optimiser update
            optimiser.update_params()
    

# TODO: make this better
def test_network(network: NetworkDense, test_data: tuple[cp.ndarray, cp.ndarray]):

    (x_test, y_test) = test_data

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
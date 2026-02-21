'''

'''

import cupy as cp
import numpy as np

from neural.core.network import NetworkDense
from neural.loss.cce import CCE
from neural.optimisers.sgd import SGD

from neural.core.datasets import get_mnist_batches


def train_network(network: NetworkDense, training_data: tuple[cp.ndarray, cp.ndarray], hyperparams: dict) -> NetworkDense:
    '''
    Train a network on the given data using the settings in hyperparams.

    Keyword Args:
        :param network: Network to be trained.
        :param training_data: Data to train network on.
        :param hyperparams: Hyperparam dict containing epochs, l_rate, w_decay, batch_size.

    Returns:
        Trained network.

    '''

    loss = CCE()
    optimiser = SGD(network, hyperparams['l_rate'], hyperparams['w_decay'])
    
    (x_train, y_train) = training_data

    batch_size = hyperparams['batch_size']
    for epoch in range(1, 1+hyperparams['epochs']):
        
        print(f'Epoch {epoch}...')

        # Dynamically slice list to allow for imperfect divisor batch_size
        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            # fwd pass (notice loss calculation can be skipped)
            network.fwd(x_batch)

            # bwd pass
            loss.bwd()
            network.bwd(chain_dvals=loss.input_dvals, true_classes=y_batch)

            # optimiser update
            optimiser.update_params()

    return network
    

# TODO: make this better
def test_network(network: NetworkDense, test_data: tuple[cp.ndarray, cp.ndarray]):
    '''
    Test a network on the given testing data, returning its average accuracy & loss.
    '''

    (x_test, y_test) = test_data
    n_samples = x_test.shape[0]

    loss = CCE()
    
    # Fwd pass
    network.fwd(x_test)
    loss.calculate(network.probs, y_test)
    
    # Average loss over each sample
    avg_loss = loss.outputs / n_samples

    # Get the most confident class & compare to true class for acc
    # Then avg. over samples
    pred = cp.argmax(network.probs, axis=1)
    pred_total = cp.where(pred == y_test)
    avg_accuracy = 100 * (pred_total / n_samples)
    
    print(f'Average loss on test set: {avg_loss}')
    print(f'Average accuracy on test set: {avg_accuracy}')

    return avg_accuracy, avg_loss


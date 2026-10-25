'''

'''

# 3rd party
import cupy as cp

# Internal
from mnist_from_scratch.neural import NetworkDense, CCE, SGD


def train_network(network: NetworkDense, training_data: tuple[cp.ndarray, cp.ndarray], hyperparams: dict) -> NetworkDense:
    '''
    Train a network on the given data using the settings in hyperparams.

    Keyword Args:
        :param network: Network to be trained.
        :param training_data: Data to train network on.
        :param hyperparams: Hyperparam dict containing epochs, learning_rate, weight_decay, batch_size.

    Returns:
        Trained network.

    '''

    loss = CCE()
    optimiser = SGD(network, hyperparams['learning_rate'], hyperparams['weight_decay'])
    
    (x_train, y_train) = training_data

    batch_size = hyperparams['batch_size']
    for epoch in range(1, 1+hyperparams['epochs']):
        
        # Shuffle the data for better training
        cp.random.shuffle(x_train)
        cp.random.shuffle(y_train)

        print(f'Epoch {epoch}...')

        # Steps through each batch
        for i in range(0, x_train.shape[0], batch_size):

            # Dynamically slice list to allow for imperfect divisor batch_size
            x_batch = x_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            # fwd pass (notice loss calculation can be skipped)
            network.fwd(x_batch)

            # bwd pass
            loss.bwd(probs=network.probs, true_classes=y_batch)
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
    n_samples = y_test.shape[0]

    loss = CCE()
    
    # Fwd pass
    network.fwd(x_test)
    loss.calculate(network.probs, y_test)
    
    # Average total loss over each sample
    avg_loss = float(cp.mean(loss.outputs))

    # Calculate accuracy 
    pred_indices = cp.argmax(network.probs, axis=1) # Indices of highest probs
    
    # Array of True/False where y_test and probs line up
    bool_mask = cp.array(y_test[cp.arange(n_samples), pred_indices] == 1) 
    n_cor_pred = int(cp.sum(bool_mask)) # Sums the True (1) and False to get a total number of correct predictions

    avg_accuracy = 100 * (n_cor_pred / n_samples)
    
    # For debug, can put in log file later if required
    print(f'Average loss on test set: {avg_loss}')
    print(f'Average accuracy on test set: {avg_accuracy}')

    return avg_accuracy, avg_loss


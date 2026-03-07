import numpy as np

from tools.debugging import debug_style

from neural.core import NetworkDense
from neural.layers import LayerDense, ReLU, Softmax
from neural.core import get_mnist_batches, get_mnist_data
from neural.tests.data import gen_test_data
from neural.tests.dummy import gen_test_one_hot, NetworkDenseExample
from neural.loss import CCE
from neural.optimisers import SGD


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

        loss = CCE()
        loss.calculate(test_net.network.probs, true_classes)
        print(f"\n{debug_style(f'[Loss]: {loss.outputs}')}\n")


def manual_bwd(network: NetworkDense, loss: CCE, cor_outputs: np.ndarray) -> None:
    
    smax: Softmax = network.layers[-1] # type: ignore
    smax.bwd(loss.input_dvals, true_classes=cor_outputs)
    print(debug_style(f'Softmax local_dvals:\n{smax.local_input_dvals}\n' # type: ignore
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
    weight_decay = 0.1
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

        loss = CCE()
        loss.calculate(network.probs, cor_outputs)
        print(f"\n{debug_style(f'[Loss]: {loss.outputs}')}")

        loss.bwd()
        print(debug_style(f'[Loss Derivative] {loss.input_dvals}'))

        network.bwd(loss_dvals=loss.input_dvals, true_classes=cor_outputs)

        sgd = SGD(learning_rate=learning_rate, network=network, weight_decay=weight_decay)
        sgd.update_params()


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


def net_train_test() -> float:
    
    # For readability
    np.set_printoptions(suppress=True, precision=3)

    n_inputs = 784
    n_neurons = 64
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 0.1 # Halfed because not using traditional L2
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
    # optimiser = SGD(learning_rate=0.01, network=network)

    (x_train_batches, y_train_batches), (x_test, y_test) = get_mnist_batches(batch_size=batch_size)

    loss = CCE()
    optimiser = SGD(learning_rate=learning_rate, 
                                            network=network, weight_decay=weight_decay)

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
        
        optimiser.update_params()

        # print(debug_style(f'probs:\n{network.probs}'))

        # for i, sample in enumerate(network.probs):
        #     if np.any(sample >= 0.9):
        #         if np.where(sample >= 0.9) != np.where(true_classes[i] == 1):
        #             print(warning_style(f'Confidently incorrect guess:\n{sample}\n{true_classes[i]}'))
        #         print(f'{sample}\n{true_classes[i]}')

        # input('...')
    
    print(f'Number of one overflows in 60,000 MNIST: {network.layers[-1].one_overflows}')
    

    (new_x_train, new_y_train), (new_x_test, new_y_test) = get_mnist_data()

    network.layers[-1].debug = False
    print('printer')

    avg_acc = net_test(network, x_test=new_x_train, y_test=new_y_train)

    return avg_acc

        # optimiser.update_params()


def softmax_test():
    smax = Softmax()

    inputs = np.array([[ 19.56129821, -32.53570809, -30.15035223, 101.03722652,  31.35531259,
   6.68513487, -68.16097319,  54.38668008, -73.05622014,   2.29131153]])

    print(f'abs_inputs: {abs(inputs)}')

    smax.fwd(abs(inputs))

    print(smax.outputs)


def test_saving() -> None:
    net_params = np.load('logs//layer_params.npz')

    param_names = sorted(net_params.files)
    print(param_names)
    
    for param_set in param_names:
        print(net_params[param_set].shape)

import os
import copy

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

import neural
from neural import NetworkDense, LayerDense, ReLU, Softmax, CCE, SGD
import neural.layers.activations as activations

def test_parameter_initialisation() -> None:
    
    # Instantiate dense layer
    layer = LayerDense(64, 784)
    
    # Check the shape of weights, biases is correct
    assert layer.weights.shape == (64, 784)
    assert layer.biases.shape == (1, 64)

    # Check that the weights are randomly initialised
    assert len(np.unique(layer.weights)) > 1

    # Move to CPU first
    layer_weights_cpu = cp.asnumpy(layer.weights)

    # Check that the weights use He initialisation
    assert np.isclose(np.var(layer.weights), 2 / 784, rtol=0.1)
    
    print('Tests (1a), (1c) passed. See graph for Test (1b).')
    
    # Plot graph of weights
    plt.hist(layer_weights_cpu.flatten(), bins=50)
    plt.show()



def test_network_initialisation() -> None:

    from neural.core.base import Activation, Loss

    # Initialise two dense layers
    layer1 = LayerDense(64, 784)
    layer2 = LayerDense(64, 10)
    
    # Check that they have weights and biases
    assert hasattr(layer1, 'weights')
    assert hasattr(layer1, 'weights')
    assert hasattr(layer2, 'biases')
    assert hasattr(layer2, 'biases')
    
    # Instantiate a network using the layers and activation functions
    net = NetworkDense([layer1, ReLU(), layer2, Softmax()])
    
    # Check that the layers are actually in the network
    assert hasattr(net, 'layers')
    assert layer1 in net.layers
    
    # Check that the methods of activation functions are standardised
    relu_methods =[m for m in dir(ReLU) if not m.startswith('_')]
    softmax_methods =[m for m in dir(Softmax) if not m.startswith('_')]
    assert set(relu_methods) == set(softmax_methods)

    # Check that activation functions inherit from a base Activation class
    assert Activation in ReLU.__bases__ and Activation in Softmax.__bases__

    # Check that the single loss function has a method
    cce_methods = [m for m in dir(CCE) if not m.startswith('_')]
    assert len(cce_methods) > 0

    # Check that the loss function inherits from a base Loss class
    assert Loss in CCE.__bases__

    print('Tests (2a), (2b), (2c), (2d) passed.')


def test_fwd_propogation() -> None:
    
    # Initialise two dense layers
    layer1 = LayerDense(64, 784)
    layer2 = LayerDense(10, 64)

    # Random inputs
    dummy_inputs = cp.random.randn(32, 784)
    
    # Forward to each layer
    layer1.fwd(dummy_inputs)
    layer2.fwd(layer1.outputs)
    
    # Check that second layer has outputs
    assert layer2.outputs is not None
    
    # Do the same using the Network class
    net = NetworkDense([layer1, ReLU(), layer2, Softmax()])
    net.fwd(dummy_inputs)
    
    # Check that both ways result in the same outputs
    assert np.array_equal(net.probs, net.layers[-1].outputs)
    
    # Check that the shape of relu's outputs are the same as its inputs
    relu = ReLU()
    relu.fwd(layer1.outputs)
    assert relu.outputs.shape == layer1.outputs.shape
    
    # Check that the network produces outputs, 
    # and of the correct shape and values
    assert net.probs.shape == (32, 10)
    assert np.all((net.probs >= 0) & (net.probs <= 1))

    assert net.conf_scores.shape == (32, 10)
    assert cp.all((net.conf_scores >= 0) & (net.conf_scores <=100))

    # Check that there at least 3 activation functions with a fwd() method
    from neural.layers import activations
    count = 0
    for name in activations.__all__:
        cls = getattr(activations, name)
        
        # Check if cls has overwritten parent implementation of fwd()
        if cls.__bases__[0].fwd != cls.fwd:
            count += 1

    assert count >= 3

    print('Tests (3a), (3b), (3c), (3d), (3e), (3f) passed.')


def test_backpropogation() -> None:
    
    # Initialise layer and fwd dummy inputs
    layer = LayerDense(64, 784)
    dummy_inputs = cp.random.randn(32, 784)
    layer.fwd(dummy_inputs)
    
    # bwd dummy dvals
    dummy_dvalues = cp.random.randn(32, 64)
    layer.bwd(dummy_dvalues)
    
    # Check that network has weight and bias dvals
    assert hasattr(layer, 'weight_dvals') and layer.weight_dvals is not None
    assert hasattr(layer, 'bias_dvals') and layer.bias_dvals is not None
    
    # Check that relu has dvals after being bwded to 
    relu = ReLU()
    relu.fwd(layer.outputs)
    relu.bwd(dummy_dvalues)
    assert hasattr(relu, 'input_dvals') and relu.input_dvals is not None
    
    # Check that final activation (Softmax) has dvals after being bwded to
    final_activation = Softmax()
    final_activation.fwd(cp.random.randn(32, 10))
    final_activation.bwd(cp.random.randn(32, 10), cp.eye(32, 10))
    assert hasattr(final_activation, 'input_dvals') and final_activation.input_dvals is not None
    
    # Initialise full network and bwd to it
    net = NetworkDense([LayerDense(64, 784), ReLU(), LayerDense(10, 64), Softmax()])
    net.fwd(dummy_inputs)
    net.bwd(chain_dvals=cp.random.randn(32, 10), true_classes=cp.eye(32, 10))
    
    # Check that full network has dvals
    for lyr in net.layers:
        if isinstance(lyr, LayerDense):
            assert hasattr(lyr, 'weight_dvals') and lyr.weight_dvals is not None
            assert hasattr(lyr, 'bias_dvals') and lyr.bias_dvals is not None
        
        # Activation funcs
        assert hasattr(lyr, 'input_dvals') and lyr.input_dvals is not None

    print('Tests (4a), (4b), (4c), (4d), (4e) passed.')

    # Check that there at least 3 activation functions with a bwd() method
    count = 0
    for name in activations.__all__:
        cls = getattr(activations, name)

        # Check if the class has overwritten the parent class implementation
        if cls.__bases__[0].bwd != cls.bwd:
            count +=1

    assert count >= 3, "Test (4f) failed, there are less than 3 activation functions" \
                       "which can be backwarded to"


def test_learning() -> None:

    # Random
    inputs = cp.random.randn(32, 784)
    
    # Random correct classes also
    y_true = cp.zeros((32, 10))
    for i in range(32):
        y_true[i, int(cp.random.randint(0, 10))] = 1.0

    # Instantiate and fwd to net
    net = NetworkDense([LayerDense(64, 784), ReLU(), LayerDense(10, 64), Softmax()])
    net.fwd(inputs)
    
    # Calc loss and dvals
    loss_func = CCE()
    loss_func.calculate(net.probs, y_true)
    loss_func.bwd(probs=net.probs, true_classes=y_true)
    
    # Bwd to net
    net.bwd(chain_dvals=loss_func.input_dvals, true_classes=y_true)
    
    # Copy weights & biases before updating
    old_weights = cp.copy(net.layers[0].weights)
    old_biases = cp.copy(net.layers[0].biases)
    
    # Update weights & biases from net
    optimiser = SGD(network=net, learning_rate=0.01, weight_decay=0.0)
    optimiser.update_params()
    
    # Check that weights & biases have changed
    try:
        assert not cp.array_equal(net.layers[0].weights, old_weights)
        print("Test (5a) Passed")
    except AssertionError:
        print("Test (5a) Failed: Weights unchanged after update.")

    try:
        assert not cp.array_equal(net.layers[0].biases, old_biases)
        print("Test (5a) Passed.")
    except AssertionError:
        print("Test (5a) Failed: Biases unchanged after update")

    # Instantiate two nets with same params
    net1 = NetworkDense([LayerDense(10, 784), Softmax()])
    net2 = NetworkDense([LayerDense(10, 784), Softmax()])
    net2.layers[0].weights = cp.copy(net1.layers[0].weights)
    net2.layers[0].biases = cp.copy(net1.layers[0].biases)
    
    # Store initial weights
    old_w1 = cp.copy(net1.layers[0].weights)
    
    # Full fwd and bwd pass to net 1
    net1.fwd(inputs)
    loss1 = CCE()
    loss1.bwd(probs=net1.probs, true_classes=y_true)
    net1.bwd(chain_dvals=loss1.input_dvals, true_classes=y_true)
    
    # fwd, bwd to net 2
    net2.fwd(inputs)
    loss2 = CCE()
    loss2.bwd(probs=net2.probs, true_classes=y_true)
    net2.bwd(chain_dvals=loss2.input_dvals, true_classes=y_true)

    # Update net 1
    opt1 = SGD(net1, 0.01, 0.0)
    opt1.update_params()
    diff1 = float(cp.mean(cp.abs(net1.layers[0].weights - old_w1)))
    
    # Update net 2
    opt2 = SGD(net2, 0.1, 0.0)
    opt2.update_params()
    diff2 = float(cp.mean(cp.abs(net2.layers[0].weights - old_w1)))
    
    # Check that bigger learning rate results in bigger update
    try:
        assert diff2 > diff1
        print(f"Test (5b) Passed.")
    except AssertionError:
        print(f"Test (5b) Failed: Higher LR did not result in larger parameter shift")

    # Generate bad and good network probs    
    dummy_good = cp.zeros((32, 10))
    dummy_bad = cp.full((32, 10), 0.1)
    for i in range(32):
        dummy_good[i, 1] = 1.0
        
    # And fake true classes
    y_true_cce = cp.zeros((32, 10))
    for i in range(32):
        y_true_cce[i, 1] = 1.0

    # Calc loss both dummy network probs
    loss_func.calculate(dummy_good, y_true_cce)
    good_loss = float(cp.mean(loss_func.outputs))
    loss_func.calculate(dummy_bad, y_true_cce)
    bad_loss = float(cp.mean(loss_func.outputs))
    
    # Check that loss is less for good probs
    try:
        assert good_loss < bad_loss
        print(f"Test (5c) Passed.")
    except AssertionError:
        print("Test (5c) Failed: Loss function did not correctly differentiate prediction accuracy")

    # Check that there is more than one loss func
    import neural.loss
    loss_funcs = [
        name for name in dir(neural.loss) 
        if isinstance(getattr(neural.loss, name), type) and hasattr(getattr(neural.loss, name), 'calculate')
    ]
    try:
        assert len(loss_funcs) > 1
        print(f"Test (5d) Passed")
    except AssertionError:
        print("Test (5d) Failed: Fewer than two loss functions found.")

    # Check if there is an activation function with at least one learnable parameter
    # By checking if it has a non-empty kwargs dict attr
    learnable_found = False
    learnable_updates = False
    for name in dir(activations):
        cls = getattr(activations, name)
        if isinstance(cls, type) and hasattr(cls, 'fwd'):
            try:
                inst = cls()
                
            # If cls requires param(s), it will raise TypeError
            except TypeError:
                learnable_found = True

                # If act func also has bwd
                if cls.__bases__[0].bwd != cls.bwd:
                    learnable_updates = True
    
    if learnable_found:
        if not learnable_updates:
            print("Test (5e) Partial: Activation function has learnable parameter, but does not update it during training.")
        else:
            print("Test (5e) passed.")
    else:
        print("Test (5e) failed: No activation function with learnable parameter")

    # Check there is more than one optimiser
    opt_dir = os.path.join(os.path.dirname(neural.__file__), 'optimisers')
    opt_files = [f for f in os.listdir(opt_dir) if f.endswith('.py') and f != '__init__.py']
    try:
        assert len(opt_files) > 1
        print(f"Test (5f) Passed.")
    except AssertionError:
        print("Test (5f) Failed: Fewer than two optimisers found.")


def test_parameter_saving() -> None:

    # Will be needed 
    from training import save_model_params
    import json

    # Init net
    net1 = NetworkDense([LayerDense(64, 784), ReLU(), LayerDense(10, 64), Softmax()])
    
    # Give dense layers random biases also
    net1.layers[0].biases = cp.random.randn(1, 64)
    net1.layers[2].biases = cp.random.randn(1, 10)

    # Dummy metadata and path
    test_path = 'test_model_saving.npz'
    dummy_training_metadata = {'epochs': 50, 'learning_rate': 0.01}

    # Save net to dummy path w metadata
    save_model_params(net1, test_path, dummy_training_metadata)

    # Check that the file was created
    assert os.path.exists(test_path)

    # Load container for file
    raw_net_cont = np.load(file=test_path)
    raw_metadata_json_str = str(raw_net_cont['metadata'])
    metadata = json.loads(raw_metadata_json_str)

    # Load new net from old net param file
    net2 = NetworkDense.load_pretrained(raw_net_cont, metadata['layers'])

    # Check that layer numbers of both nets are the same
    assert len(net1.layers) == len(net2.layers)

    # Compare both layers
    for l1, l2 in zip(net1.layers, net2.layers):

        # Check that the type of each layer is the same
        assert l1.__class__.__name__ == l2.__class__.__name__
        
        # Check that the values of params are the same
        if isinstance(l1, LayerDense):
            assert cp.array_equal(l1.weights, l2.weights)
            assert cp.array_equal(l1.biases, l2.biases)
            assert cp.count_nonzero(l2.biases) > 0

    print('Tests (6a), (6b) passed.')

def main() -> None:
    # test_parameter_initialisation()
    test_network_initialisation()
    # test_fwd_propogation()
    # test_backpropogation(k)
    # test_learning()
    # test_parameter_saving()


if __name__ == '__main__':
    main()
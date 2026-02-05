import numpy as np

from neural import NetworkDense, save_params, LayerDense, ReLU, Softmax


net = NetworkDense(
    [
        LayerDense(10, 10),
        ReLU(),
        LayerDense(10, 10),
        Softmax()
    ]
)

save_params(net, path='logs//a_test_params')

params = np.load('logs//a_test_params.npz')
print(params['layer0_biases'])
'''
The master file from which other functions and classes are called & instantiated. 
Created: Sprint 1
Last Modified: Sprint 2
'''

import numpy as np

from neural import NetworkDense
from neural.tests import NetworkDenseExample, gen_test_one_hot
from tools.debugging import debug_style
from neural.layers.loss import CategoricalCrossEntropy

def main() -> None:
    
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
        print(debug_style(f'[FINAL OUTPUTS]:\n{test_net.network.conf_scores}'))
        print(debug_style(f'[Sum across neurons][axis 1]:\n{np.sum(test_net.network.conf_scores, axis=1)}'))

        true_classes = gen_test_one_hot(layer_arch=test_net.layer_arch, batch_size=batch_size) # pyright: ignore[reportArgumentType]

        loss = CategoricalCrossEntropy()
        loss.calculate(test_net.network.conf_scores, true_classes)
        print(f"\n{debug_style(f'[Loss]: {loss.outputs}')}\n")

if __name__ == '__main__':
    main()

# main.py




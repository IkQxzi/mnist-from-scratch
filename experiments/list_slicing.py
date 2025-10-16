list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

print(f'list: {list}')
print(f'sliced list (list[1:]): {list[1:]}')

for i, item in enumerate(list[1:]):
    print(i, item)


import numpy as np
class LayerDense: 

    def __init__(self, n_neurons, n_inputs) -> None:

        
        self.weights = np.random.randn(n_neurons, n_inputs)#


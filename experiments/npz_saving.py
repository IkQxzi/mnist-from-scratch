import numpy as np

metadata = {
		'layers': [
                {'type': 'LayerDense', 'args': {'n_neurons': 784, 'n_inputs': 784}, 'params': ['dense1_weights', 'dense1_biases']},
        		{'type': 'ReLU'},
            ],
		'epochs': 30
        # ...
	}

np.savez(file='logs//meta_save.npz', metadata=metadata)
from neural_network.core import NetworkNeural, LayerDense

def main() -> None:
    
    n_layers: int = 3
    n_layer_neurons = [4, 3, 3]
    n_network_inputs = 3

    network = NetworkNeural(n_layers, n_layer_neurons, n_network_inputs)



if __name__ == '__main__':
    main()



# main.py
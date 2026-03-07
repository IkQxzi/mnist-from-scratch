import time

from neural import NetworkDense, LayerDense, ReLU, Softmax, CCE, SGD
from neural.core.datasets import get_mnist_data
from training import train_network, test_network, save_model_params

def main() -> None:

    HYPERPARAMS = {
        'epochs': 30,
        'learning_rate': 0.01,
        'weight_decay': 1e-4,
        'batch_size': 32
    }

    N_NEURONS = [784, 64, 10]

    (x_train, y_train), (x_test, y_test) = get_mnist_data()

    net = NetworkDense(
        [
            LayerDense(784, N_NEURONS[0]),
            ReLU(),
            LayerDense(N_NEURONS[1], N_NEURONS[0]),
            ReLU(),
            LayerDense(N_NEURONS[2], N_NEURONS[1]),
            Softmax()
        ]
    )

    net = train_network(net, (x_train, y_train), HYPERPARAMS)
    avg_acc, avg_loss = test_network(net, (x_test, y_test))

    training_metadata = HYPERPARAMS | ({
        'avg_acc': avg_acc,
        'avg_loss': avg_loss
    })

    timestamp = time.strftime('%Y-%m-%d_%H.%M')
    save_path = rf'C:\Users\ikram\Documents\Schoolwork\compsci\NEA 2026\src\training\models\new_{avg_acc}_{timestamp}'

    save_model_params(net, save_path, training_metadata)


if __name__ == '__main__':
    while True:
        main()

        


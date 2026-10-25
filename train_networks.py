from datetime import datetime

# Internal
from mnist_from_scratch.config import MODEL_DIR
from mnist_from_scratch.neural import LayerDense, NetworkDense, ReLU, Softmax
from mnist_from_scratch.neural.core import get_mnist_data
from mnist_from_scratch.training import save_model_params, test_network, train_network


def main() -> None:

    HYPERPARAMS = {
        "name": f"ikqnet{datetime.now().isoformat(timespec='seconds')}",
        "epochs": 999,
        "learning_rate": 0.01,
        "weight_decay": 1e-4,
        "batch_size": 64,
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
            Softmax(),
        ]
    )

    net = train_network(net, (x_train, y_train), HYPERPARAMS)
    avg_acc, avg_loss = test_network(net, (x_test, y_test))

    training_metadata = HYPERPARAMS | ({"avg_acc": avg_acc, "avg_loss": avg_loss})

    # Replace colons in timestamp with dots to prevent errors with Windows file mgmt.
    timestamp = (datetime.now().isoformat(timespec="seconds")).replace(":", ".")
    save_path = (
        MODEL_DIR
        / f"{HYPERPARAMS['name']}_{avg_acc:.2f}_{avg_loss:.4f}_{len(net.layers)}_{timestamp}.npz"
    )

    save_model_params(net, save_path, training_metadata)


if __name__ == "__main__":
    main()

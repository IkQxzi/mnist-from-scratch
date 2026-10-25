from mnist_from_scratch.neural.core.datasets import get_mnist_batches


def batch_test() -> None:
    (x_train_batches, y_train_batches), (x_test, y_test) = get_mnist_batches(32)

    print(f"x_train_batches shape: {x_train_batches.shape}")
    print(f"y_train_batches shape: {y_train_batches.shape}")
    print(f"x_train_batches:\n {x_train_batches[0, 0, :28]}")
    print(f"y_train_batches:\n {y_train_batches[0, :5]}")

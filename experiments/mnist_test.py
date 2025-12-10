import numpy as np
from keras.datasets import mnist

arr = np.array([34, 4, 4])
print(len(arr.shape), arr.size)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_empty = np.zeros(shape=(y_train.size, 10))


# from the 0th to the last arr,
# fill each subarr with the value of y_train at that subarr index
y_empty[np.arange(y_empty.shape[0]), y_train] = 1

print(y_empty[:5])

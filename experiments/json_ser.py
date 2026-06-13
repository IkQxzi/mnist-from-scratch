import json

from neural import LayerDense

dense1 = LayerDense(1, 1)
dense2 = LayerDense(2, 2)

dense_dict = {
    'dense1': dense1.__class__.__name__,
    'dense2': dense2.__class__.__name__
}

print(json.dumps(dense_dict))


arr = [0, 1, 2, 3, 4]
print(json.dumps(arr))
print(type(arr))
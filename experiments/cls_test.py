import neural.layers.dense

dense_type = getattr(dense, "LayerDense")
print(type(dense_type))
print(dense_type)

my_layer = dense_type()


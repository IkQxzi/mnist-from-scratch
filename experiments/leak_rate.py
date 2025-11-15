
leak_rate = 3
layer_arch = (256, 'ReLU', 16, 'LeakyReLU', 10, 'Softmax')

if leak_rate and any(txt.lower() in ['leakyrelu', 'leaky_relu'] for txt in layer_arch):
    raise NotImplementedError('Leaky ReLU activation requires leak_rate to be specified.')

print('Success!')
import numpy as np

NET_FILE = r"C:\Users\ikram\Documents\Schoolwork\compsci\NEA 2026\logs\layer_params_94.37_0.005.npz"

net = np.load(NET_FILE)
print(net)
print(net['layer1_weights'])
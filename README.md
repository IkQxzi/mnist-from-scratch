# mnist-from-scratch: A neural network framework from scratch, with a GUI to draw digits
A framework to build neural networks trained on MNIST, made from scratch using CuPy (no ML frameworks), plus a frontend to draw digits and have trained networks guess them.

![demo](demos/demo1.gif)

**Status:** Complete (Sept. 2025 - Apr. 2026), no longer maintained.

## What it does
1. Train a network with various settings:
    - Network config: Number of neurons in each layer, order of layers
    - Layer & training config: Dense, ReLU, Softmax output, CCE loss, SGD optimiser
    - Hyperparameters: epoch, learning rate, weight decay, batch size
2. Use the GUI to interact with the network:
    - Select a trained network
    - Input a digit, either drawing using the mouse, or an image upload
    - Scan the digit and view the network's outputs

## How it works
- The API attempts to be PyTorch-like (but not identical)
- Neural layers are dense, with He initialisation
- Activation layers are either ReLU or Softmax
- Mini-batch SGD w/ CCE (L2 weight decay)
- Models + metadata saved as singular .npz files
- Training and inference both run on the GPU with CuPy
- GUI built with CustomTkinter
- Images are pre-processed to try to mirror MNIST format as best as possible:
    - Digits are cropped
    - They are centred by their centre of mass
    - They are then padded to 28x28 

## Usage
**Requirements:** NVIDIA GPU + Python, uv package manager
- No CUDA toolkit required (uv installs headers)

```Shell
uv sync 

uv run train_networks.py  # saved to src/mnist_from_scratch/training/models
uv run python main.py  # runs gui app
```

## Results
Networks trained using this framework using pretty standard settings get around 97.5% accuracy on the MNIST test dataset.
- Up from 95% after ReLU gradient bug fixed
**Settings used:** 784 (input) -> 784 -> 64 -> 10 neurons, 30 epochs, batch size 64, learning rate 0.01, weight decay 1e-4

## Retrospective
### Why going in semi-blind was not a great idea
I built this without studying the prerequisite maths intensely (Jacobians and such). Therefore, it took a lot of trial-and-error to get all the shapes correct, everything working properly.

All of the derivations (the gradients) were done by hand, which was a good learning exercise. However, I wish that I had looked at the algorithms actually used widely, because I never even learnt what autograd is or how to calculate the derivatives of Softmax cleanly (read its implementation in `activations.py`, and the way it's called in `core/network.py` to understand what I mean).

I believe it would have been beneficial to have worked a little with PyTorch, to familiarise myself with how a good machine-learning library should behave. 

### Some bugs
One of the mistakes I made while building this project was running `cp.random.shuffle()` on both the input data, and true classes separately, thus shuffling them independently of one another and resulting in randomly-guessing networks.

### Code Quality
This project was created back when I did not know how to use industry standard tooling or conventions (`uv` is a recent addition). 
Thus, I did not follow best practices, like using `pytest`, `logging`, a formatter like Ruff, and proper git hygiene.
The modularity and maintainability is questionable at best - saving and reading model functions are heavily coupled, as well as network loading, getting network hyperparams. It would be easier to build from scratch (which I may well do in future).

## Acknowledgements/Inspirations
### Theory
I watched (and greatly enjoyed!) [the video series on neural networks by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi); that was my main point of reference for how neural networks actually work.
I learnt basic multivariable calc. & linear algebra from 3b1b also.
I much enjoyed the visual explanations in ["How models learn" by Welsh Labs](https://www.youtube.com/watch?v=NrO20Jb-hy0), they gave me great understanding of local minima, backpropogation, etc.

### Implementation
I was heavily inspired by the video series on building neural networks from scratch in Python by Sentdex ([book](https://nnfs.io/) [video series](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)), at first following along, and then deviating after a while to work completely on my own. You will see some resemblance of his code in mine.




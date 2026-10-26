# mnist-from-scratch: A neural network frameowrk from scratch, with a GUI to draw digits
A framework to build neural networks trained on MNIST, made from scratch using Cupy (no ML frameworks)
Plus, a frontend to draw digits and have trained networks guess them.

![demo](demos/demo1.gif)

**Status:** Complete (Sept. 2025 - Apr. 2026), no longer maintained.

## What it does
1. Train a network with various settings:
    - Network config: Number of neurons in each layer, order of layers
    - Layer & training config: ReLU or Dense, Softmax output, CCE loss, SGD optimser
    - Hyperparameters: epoch, learning rate, weight decay, batch size
2. Use the GUI to interact with the network:
    - Select a trained network
    - Input a digit, either drawing using the mouse, or an image upload
    - Scan the digit and view the network's outputs

## How it works
- The API attempts to be PyTorch-like (but not identical)
- Neural layers are dense, with He initialisation
- Activation layers are either ReLU or Softmax
- CCE loss, SGD optimiser
- Mini-batch SGD w/ L2 weight decay (technically doubled)
- Models + metadata saved as singular .npz files
- Training and inference both run on the GPU with CuPy
- GUI bulit with CustomTkinter

## Usage
**Requirements:** NVIDIA GPU + Python, uv package manager

```Shell
uv sync 

uv run train_networks.py  # saved to src/mnist_from_scratch/training/models
uv run python main.py  # runs gui app
```

## Retrospective
**Why going in semi-blind was not a great idea**
I built this without studying the prerequisite maths intensely (Jacobeans and such). Therefore, it took a lot of trial-and-error to get all the shapes correct, everything working properly.
All of the derivations (the gradients) were done by hand, which was a good learning exercise. However, I wish that I had looked at the algorithms actually used widely, because I never even learnt what autograd is, how to calculate the derivatives of Softmax cleanly (read its implementation in `activations.py`, and the way its called in `core/network.py` to understand what I mean).
I believe it would have been more beneficial to have worked with PyTorch first, to familiarise myself with how a good machine-learning library should actually behave. 

**Code Quality**
This project was created back when I did not know how to use industry standard tooling or conventions (`uv` is a recent addition). 
Thus, I made some junior mistakes:
- Not writing typical tests with `pytest` (the `tests/` directory is full of informal tests)
- Using strings with `open()` everywhere, instead of `Path` objects from `pathlib`
- Using raw `print()` statements instead of using Python's built-in `logging` module
- Not using an industry-standard formatter like Ruff
- Committing all of my changes directly to the `main` branch
Not to mention, the modularity and maintainability is a complete mess - saving and reading model functions are heavily coupled, as well as network loading, getting network hyperparams, it's a real mess. It would be easier to build from scratch (which I may well do in future).
But worst of all, after training some networks for the GUI, the networks were then learning noise because I made an egregious error, running `cp.shuffle()` on both the input data, and true classes separately, thus shuffling them independently of one another and causing networks to learn complete noise.

## Acknowledgements/Inspirations
**Theory**
I watched (and greatly enjoyed!) [the video series on neural networks by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi); that was my main point of reference for how neural networks actually work.
I learnt basic multivariable calc. & linear algebra from 3b1b also.
I much enjoyed the visual explanations in ["How models learn" by Welsh Labs](https://www.youtube.com/watch?v=NrO20Jb-hy0), they gave me great understanding of local minima, backpropogation, etc.

**Implementation**
I was heavily inspired by the video series on building neural networks from scratch in Python by Sentdex ([book](https://nnfs.io/) [video series](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)), at first folowing along, and then deviating after a while to work completely on my own. You will see some resemblance of his code in mine.




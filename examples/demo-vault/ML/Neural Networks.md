# Neural Networks

A neural network is a stack of layers, each computing a weighted sum of its
inputs followed by a non-linearity. With enough layers and units it can
approximate almost any function.

## Multilayer perceptron

The simplest architecture: fully connected layers, e.g. 784 -> 128 -> 10 for
MNIST digits. Non-linear activations (ReLU, GELU) are what let depth add
expressive power; without them the whole stack collapses to one linear map.

## Training

The network's error is measured by a loss function. **Backpropagation** applies
the chain rule backwards through the layers to get the gradient of the loss
with respect to every weight, and an optimiser uses those gradients to update
the weights.

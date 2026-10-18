# Backpropagation

Backprop is just the chain rule applied efficiently to a computation graph. A forward pass computes activations layer by layer and caches them; the backward pass walks the graph in reverse, multiplying local derivatives to get dLoss/dParameter for every weight in one sweep. The cost is roughly 2x the forward pass.

Those gradients are what gradient descent consumes to update the weights; backprop computes them, the optimiser decides what to do with them.

Practical problems:
- **Vanishing gradients**: sigmoids/tanh squash derivatives below 1, so after many layers the signal dies. ReLU, residual connections and careful initialisation (He/Xavier) help.
- **Exploding gradients**: the opposite, common in RNNs. Gradient clipping fixes it.
- Normalisation layers (batch norm, layer norm) keep activations in a healthy range so gradients flow better.

Autograd frameworks (PyTorch, JAX) build the graph for you, so you rarely write backward passes by hand anymore.

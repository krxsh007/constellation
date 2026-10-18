# Gradient Descent

Gradient descent minimises a loss by repeatedly stepping the parameters in the
direction of the negative gradient.

- **Batch** gradient descent uses the whole dataset per step: accurate but slow.
- **Stochastic** gradient descent (SGD) uses one example or a mini-batch per
  step. The noise is a feature: it helps escape sharp minima.
- Momentum and Adam keep running averages of past gradients to smooth the path.

## Learning rate

The single most important hyperparameter. Too high and the loss diverges; too
low and training crawls. Warmup followed by cosine decay is a solid default
for large models.

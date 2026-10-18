# Gradient Descent

The workhorse optimiser for almost every model I train. Start from random weights, compute the gradient of the loss with respect to each parameter, and step a little in the opposite direction:

    w <- w - lr * dL/dw

Variants worth remembering:
- **Batch GD** uses the whole dataset per step: stable but slow.
- **Stochastic GD (SGD)** uses one example: noisy, but the noise helps escape sharp minima.
- **Mini-batch** is what everyone actually does (32-512 examples).
- **Momentum / Adam** keep a running average of past gradients so the step direction is smoother.

The step size (learning rate) is the single most important hyperparameter. Too large and the loss diverges, too small and training crawls; see my note on learning rate schedules. The gradients themselves come from backpropagation through the network.

Loss surfaces of deep nets are non-convex, so there is no guarantee of a global minimum, but in practice wide flat minima generalise well.

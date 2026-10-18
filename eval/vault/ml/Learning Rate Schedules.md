# Learning Rate Schedules

A fixed learning rate is rarely optimal for gradient descent. Early in training you want big steps; near a minimum you want small ones so you settle instead of bouncing around.

Common schedules:
- **Step decay**: divide the rate by 10 at fixed epochs (classic ResNet recipe).
- **Cosine annealing**: smooth decay following half a cosine, optionally with warm restarts.
- **Linear warmup**: start near zero and ramp up over the first few thousand steps. Transformers basically require this, because Adam's variance estimates are unreliable at the start.
- **One-cycle**: ramp up then down, popularised by fast.ai.

Batch normalisation made much higher learning rates usable, which is part of why it sped training up so dramatically.

Rule of thumb: run an LR range test (increase the rate exponentially over one epoch and plot loss) and pick a value just below where the loss starts to explode.

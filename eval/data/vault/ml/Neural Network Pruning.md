# Neural Network Pruning

Pruning removes weights, neurons or whole attention heads from a trained network to make it smaller and faster, ideally with little loss in accuracy.

- **Magnitude pruning**: cut the weights with the smallest absolute value, fine-tune, repeat. Simple and surprisingly strong.
- **Structured pruning**: remove entire channels or heads so the speed-up is real on hardware (unstructured sparsity needs special kernels).
- **Lottery ticket hypothesis**: a dense net contains a sparse sub-network that, reset to its original initialisation, trains to the same accuracy.

You can often prune 80-90% of the weights of a big model. Pruning is close cousin to quantisation and distillation in the model-compression toolbox.

Related idea: dropout removes units temporarily during training; pruning removes them permanently after training. Pruning can also act as a regulariser, but that is not usually why people do it.

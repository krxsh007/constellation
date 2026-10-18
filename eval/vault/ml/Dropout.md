# Dropout

During training, dropout randomly zeroes each activation with probability p (typically 0.1-0.5) and scales the survivors by 1/(1-p). At inference everything is kept.

Why it fights overfitting: no neuron can rely on any specific other neuron being present, so the network learns redundant, more robust features. It can be seen as training an exponential number of thinned sub-networks that share weights and averaging them at test time, a cheap ensemble.

Notes:
- Usually applied after fully-connected layers and to attention weights in transformers; less useful in conv layers (spatial dropout exists for that).
- Interacts awkwardly with batch norm, because the variance seen at train and test time differs.
- **MC dropout**: keep dropout on at inference and sample several predictions to get an uncertainty estimate.

It is a regulariser, so if the model is underfitting, lower p or remove it.

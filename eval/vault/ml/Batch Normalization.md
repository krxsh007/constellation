# Batch Normalization

Batch norm normalises each layer's activations using the mean and variance computed over the current mini-batch, then applies a learned scale (gamma) and shift (beta). It was introduced in 2015 to reduce "internal covariate shift", though the real reason it works is still debated (smoother loss landscape is the current favourite explanation).

Effects:
- Lets you train with much higher learning rates and be less careful about initialisation.
- Acts as a mild regulariser, because each example's normalisation depends on the other examples in the batch; some people drop dropout when using it.
- Behaves differently at train and inference time: at inference you use running averages of the batch statistics. Forgetting `model.eval()` is a classic bug.

Downsides: small batch sizes give noisy statistics, and it is awkward for sequence models. That is why transformers use **layer norm** instead, which normalises across features within a single example.

# Attention Mechanism

Attention lets a model compute a weighted average of a set of values, where the weights depend on how well a query matches each key:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

The sqrt(d_k) scaling stops the dot products growing with dimension and saturating the softmax.

History: Bahdanau attention (2014) was added on top of RNN encoder-decoders for translation so the decoder could "look back" at any source word. Self-attention, where queries, keys and values all come from the same sequence, became the core of the transformer architecture.

**Multi-head attention** runs several attention operations in parallel with different learned projections, so different heads can track different relationships (syntax, coreference, position).

Attention maps are tempting to read as explanations, but they are not reliable evidence of what the model "uses".

# Transformers

The transformer architecture ("Attention Is All You Need", 2017) replaced
recurrence with **self-attention**: every token builds its representation as a
weighted mix of every other token, with the weights computed from learned
query/key dot products.

- Attention is permutation-invariant, so **positional encodings** are added to
  the token embeddings to tell the model about word order.
- Each block is attention followed by a position-wise feed-forward network
  (a small MLP), with residual connections and layer norm around both.
- Cost grows quadratically with sequence length, which is why long-context
  models need tricks like sliding windows or sparse attention.

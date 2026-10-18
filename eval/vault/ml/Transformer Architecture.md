# Transformer Architecture

"Attention Is All You Need" (2017) replaced recurrence entirely with self-attention. Each block is:

1. Multi-head self-attention
2. Add & layer norm (residual connection)
3. Position-wise feed-forward network
4. Add & layer norm

Because attention has no notion of order, positional encodings (sinusoidal, learned, or rotary/RoPE these days) are added to the token embeddings.

Encoder-only (BERT), decoder-only (GPT) and encoder-decoder (T5) variants all share the same block. Decoder-only models use a causal mask so a token can only attend to earlier tokens.

Why it won: every position is processed in parallel, so it scales on GPUs far better than RNNs, and long-range dependencies are one attention hop away instead of many recurrent steps. The cost is quadratic memory in sequence length, which is what all the efficient-attention papers try to fix.

Training needs learning-rate warmup and is sensitive to normalisation placement (pre-LN is more stable than post-LN).

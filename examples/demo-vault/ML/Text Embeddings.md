# Text Embeddings

An embedding model maps a piece of text to a dense vector so that texts with
similar meaning end up close together. Sentence embedding models are usually
transformer encoders whose token outputs are pooled into one vector.

Similarity is measured with **cosine similarity**: the cosine of the angle
between two vectors, ignoring their length. Many models return normalised
vectors, in which case cosine similarity equals the dot product.

Uses I care about:

- semantic search over my notes
- clustering related ideas
- suggesting links between notes that never mention each other by name

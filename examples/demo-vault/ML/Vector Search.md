# Vector Search

Finding the nearest neighbours of a query vector among millions of stored
vectors. Exact search is a linear scan; real systems use **approximate nearest
neighbour** (ANN) indexes that trade a little recall for orders of magnitude
more speed.

- **FAISS** (Facebook AI Similarity Search) is a library of index types: flat
  (exact), IVF (cluster then search a few clusters), HNSW (graph-based) and
  product quantisation for compression.
- An index is only meaningful for vectors from the same embedding model. Mixing
  models silently produces garbage neighbours.
- Dedicated vector databases add persistence, filtering and replication on top
  of the same ideas.

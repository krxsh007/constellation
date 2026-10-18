# Vector Clocks

Physical clocks drift, so distributed systems use logical clocks to reason about ordering.

- **Lamport clocks**: one counter per process; if a happened before b then L(a) < L(b), but not the converse, so they cannot detect concurrency.
- **Vector clocks**: each process keeps a vector of counters, one per process. Increment your own entry on each event, merge (element-wise max) on receive. Event a happened before b iff a's vector is <= b's in every entry; if neither dominates, the events are **concurrent**.

That concurrency detection is exactly what leaderless replicated stores need: when two replicas accepted conflicting writes for the same key, the vector clocks show the writes are siblings, and the application (or a last-writer-wins rule, or a CRDT) resolves them. Dynamo and Riak used this.

Downside: vectors grow with the number of writers, so systems prune them or use dotted version vectors.

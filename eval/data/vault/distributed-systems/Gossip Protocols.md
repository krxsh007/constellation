# Gossip Protocols

Gossip (epidemic) protocols spread information the way rumours spread: every round, each node picks a few random peers and exchanges what it knows. Information reaches all N nodes in O(log N) rounds with high probability, with no central coordinator and graceful tolerance of failures.

Uses:
- **Membership and failure detection**: SWIM, and Serf/Consul's memberlist. Nodes gossip heartbeats and suspicions.
- **Cluster metadata**: Cassandra gossips which node owns which token ranges on its ring.
- **Anti-entropy**: replicas periodically compare data (often with Merkle trees) and repair differences, so replicas converge eventually.

Trade-offs: convergence is probabilistic and eventually consistent; bandwidth grows with state size, so real systems gossip digests and pull the full data only when something differs.

# Leader Election

Many distributed systems are simpler if exactly one node coordinates: assigning work, ordering writes, holding a lock. Leader election is how the nodes agree on who that is, and how they replace it when it fails.

Approaches:
- **Bully algorithm**: highest-ID live node wins. Simple, but chatty and assumes reliable failure detection.
- **Consensus-based**: Raft builds election into the protocol using terms and randomised timeouts; Multi-Paxos elects a distinguished proposer.
- **Lease-based via a coordination service**: ZooKeeper ephemeral sequential nodes or an etcd lease; whoever holds the lease is leader.

The hard part is not picking a leader but making sure an old leader that was only *partitioned away* (not dead) stops acting as leader. Fencing tokens: every new term gets a higher number and storage rejects writes carrying a stale token.

Failure detection is always timeout-based, so there is a trade-off between fast failover and false positives during GC pauses.

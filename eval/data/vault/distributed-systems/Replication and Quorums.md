# Replication and Quorums

Replication keeps copies of the same data on several nodes for durability, availability and read throughput.

Topologies:
- **Single-leader**: all writes go to the leader, followers replay its log. Simple; failover needs leader election.
- **Multi-leader**: writes accepted in several data centres; conflicts must be resolved.
- **Leaderless** (Dynamo-style): clients write to and read from several replicas directly.

**Quorums**: with N replicas, a write waits for W acknowledgements and a read queries R replicas. If W + R > N, every read overlaps at least one replica that saw the latest write. Typical: N=3, W=2, R=2. Sloppy quorums and hinted handoff relax this during a partition to stay available, which is the CAP trade-off in practice.

Leaderless systems detect concurrent writes to the same key with version vectors (vector clocks) and use read repair and anti-entropy to converge. Consensus protocols like Raft are really just replication with a majority quorum and strict ordering.

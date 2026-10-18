# CAP Theorem and Network Partitions

Brewer's CAP theorem: when a **network partition** splits a cluster, a replicated data store must choose between **consistency** (every read sees the latest write) and **availability** (every request gets a non-error answer). You cannot give up partition tolerance in a real network, so the real choice is CP vs AP during a partition.

- **CP systems** (etcd, ZooKeeper, Spanner) refuse writes on the minority side of a partition.
- **AP systems** (Dynamo, Cassandra with low consistency levels) keep accepting writes on both sides and reconcile later.

PACELC extends it: Else, when there is no partition, you still trade Latency against Consistency.

What actually decides behaviour in practice is the replication protocol and its quorum settings: with N replicas, W + R > N gives you read-your-writes. Partitions are not rare: misconfigured switches, GC pauses and cloud AZ outages all look like partitions to the software.

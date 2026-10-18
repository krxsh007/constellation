# Database Sharding

Sharding splits one logical database into horizontal partitions (shards) that live on different machines, so writes and storage scale beyond one node.

Choosing a shard key is the big decision:
- **Range-based** (e.g. by user ID ranges): good for range scans, but prone to hot spots.
- **Hash-based**: even distribution, usually via consistent hashing so resharding moves little data; range queries become scatter-gather.
- **Directory-based**: a lookup service maps keys to shards; flexible but another component to keep available.

Costs:
- Cross-shard joins and transactions are slow or need a protocol like two-phase commit.
- Rebalancing when a shard gets too big is operationally painful.
- Each shard still needs replication for durability, so in practice you run shards x replicas.

Vitess (MySQL) and Citus (Postgres) add sharding to existing databases; Spanner and CockroachDB shard automatically into ranges.

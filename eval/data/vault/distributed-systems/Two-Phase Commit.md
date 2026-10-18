# Two-Phase Commit

2PC makes a transaction that touches several databases (or several shards of one database) commit atomically: either all participants commit or none do.

1. **Prepare**: the coordinator asks every participant to prepare. Each one writes the transaction durably and votes yes or no.
2. **Commit/Abort**: if all voted yes, the coordinator writes a commit record and tells everyone to commit; otherwise it tells everyone to abort.

The weakness: if the coordinator crashes after participants voted yes, they are stuck holding locks ("in doubt") until it comes back. 2PC is a **blocking** protocol.

Fixes: three-phase commit (assumes bounded delays, rarely used), or replicate the coordinator's decision with consensus, as in Paxos Commit or Spanner, which runs 2PC across Paxos groups so a single crash cannot block it.

XA transactions are the standard interface; they are slow, so many systems avoid cross-shard transactions entirely with careful shard keys or sagas.

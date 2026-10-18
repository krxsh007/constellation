# Raft Consensus

Raft (Ongaro & Ousterhout, 2014) is a consensus algorithm designed to be easier to understand than Paxos while providing the same guarantees. It keeps a replicated log identical on a majority of servers.

Three sub-problems:
1. **Leader election**: servers are followers, candidates or the leader. If a follower hears no heartbeat before its randomised election timeout, it becomes a candidate, increments the term and requests votes. A majority of votes makes it leader for that term.
2. **Log replication**: clients talk only to the leader, which appends entries and replicates them to followers; an entry is committed once a majority (a quorum) has stored it.
3. **Safety**: a candidate can only win if its log is at least as up to date as a majority's, so committed entries are never lost.

Raft tolerates crash faults only (f failures out of 2f+1 nodes), not malicious nodes. Used in etcd, Consul, CockroachDB and TiKV.

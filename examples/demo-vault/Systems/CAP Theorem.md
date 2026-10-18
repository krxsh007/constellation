# CAP Theorem

A distributed data store cannot simultaneously guarantee consistency,
availability and partition tolerance. Since network partitions happen whether
you like it or not, the real choice is what to give up **during** a partition:

- **CP** systems refuse some requests to stay consistent (etcd, ZooKeeper).
- **AP** systems keep answering but may return stale data (Cassandra, DynamoDB
  in its default mode).

PACELC extends the idea: even without a partition there is a trade-off between
latency and consistency.

# Raft Consensus

Raft is a consensus algorithm designed to be understandable. A cluster of
servers agrees on a single, ordered **replicated log**; each server applies the
log entries to its state machine in the same order.

- **Leader election**: servers time out at random, become candidates and ask
  for votes. A candidate with votes from a majority becomes leader for that term.
- The leader appends client commands to its log and replicates them; an entry is
  committed once a majority has stored it.
- A minority partition cannot elect a leader, so it cannot accept writes. Raft
  chooses consistency over availability.

etcd and Consul are built on Raft.

# Paxos

Lamport's Paxos is the original practical consensus protocol: a set of acceptors agree on a single value even if some of them crash and messages are delayed or lost.

Single-decree Paxos has two phases:
1. **Prepare/Promise**: a proposer picks a ballot number n and asks a majority of acceptors to promise to ignore lower ballots; they reply with any value they already accepted.
2. **Accept/Accepted**: the proposer asks them to accept a value (the highest previously accepted one, if any). Once a majority accepts, the value is chosen.

**Multi-Paxos** runs this for every slot of a replicated log and elects a stable distinguished proposer so phase 1 can be skipped most of the time; that is essentially what Raft formalises with its leader.

It only handles crash faults. Paxos is notoriously hard to implement correctly ("Paxos Made Live" from Google is a great read). It is also the basis of Paxos Commit, a non-blocking alternative to two-phase commit.

# Byzantine Fault Tolerance

A Byzantine fault is a node that behaves arbitrarily: it may lie, send different messages to different peers, or collude. The name comes from Lamport's 1982 "Byzantine Generals Problem" thought experiment about generals who must agree on a plan of attack while some of them are traitors. It has nothing to do with the actual history of the Byzantine Empire beyond the metaphor.

Key result: tolerating f Byzantine nodes requires at least **3f + 1** nodes, versus 2f + 1 for the crash faults that Paxos and Raft handle.

**PBFT** (Castro & Liskov, 1999) made BFT practical: a primary orders requests and replicas run pre-prepare / prepare / commit phases with O(n^2) messages. Like Paxos it needs a view change to replace a faulty primary.

Where it matters: blockchains (Tendermint, HotStuff), aircraft control systems, and anywhere participants do not trust each other. Inside a single company's data centre, crash-fault consensus is usually enough and much cheaper.

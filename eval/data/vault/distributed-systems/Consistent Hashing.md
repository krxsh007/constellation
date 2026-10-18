# Consistent Hashing

Problem: assign keys to N servers so that adding or removing a server moves as few keys as possible. With `hash(key) mod N`, changing N remaps almost everything.

Consistent hashing places both servers and keys on a hash ring (0 .. 2^32). A key belongs to the first server clockwise from it. Adding a server only steals keys from its clockwise neighbour; on average only K/N keys move.

Refinements:
- **Virtual nodes**: each physical server owns many points on the ring, smoothing the load and letting bigger machines take more.
- **Rendezvous (highest random weight) hashing**: an alternative with no ring.
- **Jump consistent hash**: tiny, fast, but servers can only be added/removed at the end.

It is the standard way to place data in sharded databases and caches (Dynamo, Cassandra, memcached clients). Dynamo also replicates each key on the next N-1 distinct nodes clockwise on the ring.

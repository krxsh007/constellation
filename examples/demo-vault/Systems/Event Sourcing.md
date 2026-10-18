# Event Sourcing

Instead of storing the current state of an entity, store the sequence of
events that produced it, in an **append-only log**. Current state is a fold
over the events.

- Full audit history for free.
- **Replaying** events rebuilds state, or builds new read models that did not
  exist when the events were written.
- Schema evolution of old events is the hard part; plan for upcasting.

Often paired with CQRS: writes append events, reads come from projections.

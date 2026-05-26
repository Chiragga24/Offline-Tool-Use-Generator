# ToolBench-Style Subset Note

This repository uses a curated ToolBench-style fixture instead of the full
ToolBench corpus. The choice is deliberate: the assignment is time-boxed, and
the reviewer should be able to inspect a complete end-to-end system rather than
an incomplete full-corpus ingest.

The implemented subset covers 9 representative categories:

- Finance
- Sports
- AI / ML
- Entertainment
- Travel
- Gaming
- Events
- Food / Restaurants
- Weather

The fixture contains 45 endpoints total, with intentional schema messiness so
the loader, registry enrichment, graph builder, sampler, executor, evaluator,
and repair loop all have real work to do.

Why this subset works:

- It exercises multi-step and multi-tool chains.
- It supports clear cross-domain transitions.
- It includes realistic grounding fields such as `city`, `date`, `game_id`,
  `event_id`, `hotel_id`, `restaurant_id`, and `start_time`.
- It keeps the submission reproducible and reviewable.

All endpoints are offline mocks. Actions named `create_*`, `book_*`,
`make_*`, and `add_*` simulate state transitions only. They do not call real
services, send notifications, or create live reservations.

The full rationale for the data model, graph, sampler, and diversity choices is
documented in [DESIGN.md](DESIGN.md).

# Canonical Collection

`uns collect` invokes the sole production collector. A durable
attempt claim precedes runtime construction and every external call.
Claims and terminals use complete staging writes, file fsync, atomic
no-replace publication and directory fsync. Staging is never authoritative.
A claimed seed is permanently consumed; resume closes interrupted claims as
interrupted_failure and advances to the next unclaimed planned seed.

Collection constructs complete cross-day PRE history ending at the current
speaker's turn_start. It collects one observation from each alive observer.
The current speaker's successful observation is frozen into the one Speaker
PRE Belief Handoff for speech cognition. Trained ToM predictions never enter
this path. Bounded exhausted belief or V1 failure makes the game ineligible,
with failure and partial evidence retained; there is no gameplay fallback.

A Game Bundle separates public, audit and private directories. Public records
contain no role assignment or private cognition. Restricted evidence supports
deterministic replay and canonical validation. Bundles are atomically published
and validated before success terminals reference their digest.

Development Publication uses the verified ledger and target success count,
not a caller-selected game list. It includes every canonical success at target,
validates PRE coverage and record order, freezes five whole-game folds, and
derives the Role Sidecar from restricted replay evidence. It reports excluded
attempts for complete-case selection audit. It performs no belief-target
conversion or downstream repair.

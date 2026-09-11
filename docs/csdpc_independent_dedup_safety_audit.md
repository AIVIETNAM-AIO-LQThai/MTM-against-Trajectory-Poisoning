# CSDPC Independent Deduplication Safety Audit

## Purpose

This audit was performed after the Group-2 CSDPC source-reproduction
diagnostics to test whether the observed approximately 42--45% distinct-pattern
reduction could be an artifact of the repository's shared CSDPC implementation.

The audit used an independent standalone implementation and did not call:

- src.attacks.csdpc.patterns.iter_sequence_windows
- scripts.audit_csdpc_dedup_metric_reconciliation._compute_metrics

The audit directly loaded the clean Walker2D-medium-v2 HDF5 file, concatenated
raw state and action vectors, fit sklearn KMeans, enumerated windows, performed
independently implemented consecutive-label run-length deduplication, and
counted unique raw and deduplicated sequence types.

## Frozen common settings

Dataset:
walker2d-medium-v2

Dataset SHA256:
cf00f43add04c17fdfc2958dd581dea0851b2e5bedbe6fda073758a8f841aeda

Decision unit:
concat(s_t, a_t)

Feature preprocessing:
none

KMeans:
- k = 8
- k-means++
- n_init = 10
- max_iter = 300
- tol = 1e-4
- random_state = 0
- algorithm = lloyd

Deduplication:
remove consecutive duplicate cluster labels.

Metric:
1 - (# unique deduplicated pattern types / # unique raw sequence types)

## Independent results

### L = 5, episode boundaries respected

windows: 995235
raw distinct types: 11103
deduplicated distinct types: 6389
reduction: 42.456994%

This exactly reproduces the canonical repository result.

### L = 5, episode boundaries ignored

windows: 999996
raw distinct types: 11334
deduplicated distinct types: 6549
reduction: 42.218105%

Removing episode boundaries changes the reduction only slightly.

### L = 6, episode boundaries respected

windows: 994045
raw distinct types: 26255
deduplicated distinct types: 12080
reduction: 53.989716%

### L = 6, episode boundaries ignored

windows: 999995
raw distinct types: 27037
deduplicated distinct types: 12470
reduction: 53.878019%

## Interpretation

The independently implemented L=5, episode-respecting calculation exactly
matches the repository's canonical result.

Therefore the approximately 42.457% v2 result is not explained by an error in
the repository's shared window-enumeration or metric-counting implementation.

Removing trajectory boundaries does not materially increase the reduction.

Reading the paper's u_t,...,u_{t+l} notation literally as six positions
increases the reduction to approximately 54%, but still leaves a large gap
from the source paper's statement of nearly 80%.

Thus the two principal sequence-enumeration ambiguities recoverable from the
public paper do not explain the source discrepancy.

## Claim boundary

This audit validates the internal calculation under the tested semantics.

It does not establish which unpublished statistic or preprocessing semantics
the source authors used for their nearly-80-percent statement.

No attack setting or learner configuration was selected using this audit.
No poisoned artifact was generated.
No learner was trained.

# E2 Protocol — CSDPC Sequence-Length Source Fingerprint

## Status

`POSTFINAL DIAGNOSTIC ONLY`

E2 is a new post-final experiment. It does not modify the frozen
`walker2d-final-v1` evidence base.

The frozen Group-2 conclusion remains:

```text
SOURCE-FIDELITY DISCREPANCY UNRESOLVED
```

E2 does not reopen Group 2 for attack-strength tuning.

---

## 1. Motivation

The CSDPC source defines multi-step decision sequences and the repository's
canonical independent reproduction uses fixed-length windows of five decision
units.

A later independent safety audit checked the two most obvious interpretations:

```text
actual L = 5
actual L = 6
```

for Walker2d-medium-v2, `k = 8`, attack seed `0`.

The measured distinct-pattern reductions were approximately:

```text
L = 5 -> 42.456994%
L = 6 -> 53.989716%
```

Neither recovered the source-level statement recorded by Group 2 as
approximately 80%.

Therefore E2 is not a repeat of the L=5/L=6 check.

Its purpose is to build a systematic sequence-length fingerprint under one
fully frozen pattern pipeline.

---

## 2. Scientific Question

> How does the canonical CSDPC distinct-pattern reduction statistic vary as
> actual fixed-length decision-unit windows become longer, and does the
> source-level nearly-80-percent statistic appear anywhere in a predeclared
> length range without changing any other semantics?

This is a source-fidelity diagnostic.

It is not an attack-effectiveness experiment.

---

## 3. Frozen Dataset

Dataset:

```text
walker2d-medium-v2
```

Frozen file SHA256:

```text
cf00f43add04c17fdfc2958dd581dea0851b2e5bedbe6fda073758a8f841aeda
```

The script must abort if this SHA changes.

No other D4RL version is examined in E2.

The previous v0/v1/v2 dataset-version search remains closed.

---

## 4. Frozen Decision-Unit Pipeline

For every transition:

```text
u_t = concat(s_t, a_t)
```

Feature preprocessing:

```text
none
```

KMeans:

```text
k = 8
init = k-means++
n_init = 10
max_iter = 300
tol = 1e-4
algorithm = lloyd
```

Attack seeds:

```text
0, 1, 2
```

The fitted KMeans labels for a seed are reused across every sequence length for
that seed.

---

## 5. Window Semantics

The canonical repository pipeline is retained:

```text
original completed trajectory
    -> fixed-length original-position window
    -> consecutive-label deduplication
```

Rules:

```text
stride = 1
respect episode boundaries = true
window before deduplication = true
```

No window may cross a completed-trajectory boundary.

Consecutive duplicate labels are removed only after the raw window is formed.

---

## 6. Predeclared Sequence-Length Sweep

E2 reports every integer actual decision-unit window length:

```text
L = 2, 3, 4, 5, 6, 7, 8, 9, 10
```

No length in this range may be removed after seeing the results.

For notation diagnostics, the output also reports:

```text
endpoint_span_l = L - 1
```

This is useful because sequence notation of the form:

```text
u_t, ..., u_(t+l)
```

contains `l + 1` decision units.

However:

> E2 does not assert that `l = L - 1` is the source authors' intended code
> semantics.

It is only a transparent notation mapping.

---

## 7. Primary Metric

The primary metric is exactly the frozen Group-2 canonical interpretation:

```text
distinct_pattern_reduction
=
1
-
(
    number of unique deduplicated pattern types
    /
    number of unique raw cluster-label sequence types
)
```

The implementation reuses:

```text
scripts.audit_csdpc_dedup_metric_reconciliation._compute_metrics
```

which itself uses:

```text
src.attacks.csdpc.patterns.iter_sequence_windows
```

This keeps E2 directly comparable to the frozen canonical reconciliation audit.

---

## 8. Secondary Metrics

For every seed and every sequence length, E2 also records the complete metric
set already frozen by the dedup reconciliation audit:

```text
window_count
raw_distinct_sequence_type_count
deduplicated_distinct_pattern_type_count
canonical_distinct_type_reduction_fraction
window_instance_changed_fraction
raw_distinct_sequence_type_changed_fraction
total_label_token_reduction_fraction
average_deduplicated_pattern_length
full_length_pattern_occurrence_fraction
full_length_distinct_pattern_fraction
non_equivalent_total_window_denominator_reduction_fraction
```

These secondary metrics are diagnostic only.

They must not be silently relabeled as the source paper's distinct-pattern
reduction statistic.

---

## 9. Source Reference

The Group-2 record describes the source statistic as:

```text
nearly 80%
```

E2 therefore stores:

```text
reference_fraction = 0.80
```

only as a descriptive reference line.

It is:

```text
not an acceptance gate
not an optimization target
not authorization to create a new poison setting
```

The output may report which predeclared `L` has the smallest absolute distance
to 0.80.

That report is descriptive only.

---

## 10. Historical Regression Anchors

Before accepting E2 output, seed `0` must reproduce the already-frozen
independent safety-audit values within numerical tolerance:

```text
L = 5 -> 0.42456994
L = 6 -> 0.53989716
```

These checks guard against accidental pipeline drift.

If either anchor fails, E2 must stop and the results must not be interpreted.

---

## 11. Prohibited Changes

E2 does not vary:

```text
dataset version
k
feature scaling
trajectory-boundary handling
deduplication order
window stride
rare-pattern selection
overlap semantics
candidate count
candidate-generation procedure
eta
rho
victim learner
training seed
checkpoint selection
```

E2 generates no attack artifact.

---

## 12. Interpretation Rules

### Case A — no predeclared length approaches the source reference

Conclusion:

> Sequence length alone does not reconcile the source distinct-pattern
> statistic within the tested canonical pipeline.

### Case B — one or more longer lengths approach the source reference

Conclusion:

> The distinct-pattern statistic is strongly sequence-length sensitive, and a
> longer effective window can numerically approach the source-level statistic.

This does **not** establish that the source implementation used that length.

It does **not** authorize using that length as a stronger attack.

### Case C — the curve crosses or exceeds the reference

Report the full curve.

Do not select only the crossing point.

---

## 13. Claim Boundary

Allowed:

> Under the frozen canonical Walker2d pattern pipeline, distinct-pattern
> reduction changes systematically with actual sequence length.

If supported by the results:

> A longer predeclared window length numerically approaches the source-level
> nearly-80-percent statistic.

Not allowed:

> We discovered the source authors' true sequence length.

Not allowed:

> The length closest to 80% is the correct CSDPC attack setting.

Not allowed:

> E2 reproduces source-paper attack effectiveness.

---

## 14. Output

E2 writes:

```text
experiments/postfinal_controls/e2_csdpc_sequence_length_fingerprint.json
```

No poisoned HDF5 file is written.

No learner checkpoint is written.

---

## 15. Next Decision

After E2, choose the next source-fidelity diagnostic from the pattern evidence,
not from downstream learner degradation.

If sequence length alone still does not reconcile the source fingerprint, the
next planned investigation remains candidate-generation semantics rather than
post-hoc victim-performance tuning.

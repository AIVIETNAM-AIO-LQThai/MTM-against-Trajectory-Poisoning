# CSDPC Mechanism / Fidelity Discrepancy Audit

## Status

The canonical k=8 CSDPC reproduction remains frozen.

Canonical Gate B is INCONCLUSIVE.

The predeclared k=6 and k=10 cluster-count sensitivities also failed
to produce a strong CQL degradation effect.

No canonical setting will be changed in response to these results.

## Why a mechanism audit is required

The current reproduction passes artifact-integrity and reproducibility
checks, but downstream CQL degradation is much weaker than reported by
the CSDPC source paper.

The next question is therefore upstream of the learner:

    Does the generated poisoned dataset actually collapse
    sequence-level decision-pattern coverage?

This audit uses no CQL, DT, BC, or MTM performance to select an
alternative attack configuration.

## Known source / reproduction differences

### 1. Decision-pattern deduplication

The source paper reports that deduplication reduces the number of
distinct Walker2D decision patterns by nearly 80%.

The current canonical reproduction produces an approximately 42--44%
reduction across attack seeds.

This is treated as a descriptive fidelity discrepancy, not as a
post-hoc acceptance threshold.

### 2. Perturbation candidate generation

The source attack generates multiple bounded poisoned candidates and
chooses a candidate associated with a more frequent decision pattern.

The publication does not sufficiently specify the candidate count or
candidate sampling distribution.

Canonical reproduction choice:

    num_candidates = 100
    independent seeded Uniform bounded perturbations

This choice remains frozen.

### 3. Poison-budget and overlap semantics

The source defines poisoning through low-frequency decision patterns
selected according to rho, but does not operationally specify all
transition/window overlap accounting.

Canonical reproduction choices:

    budget = floor(rho * N transitions)
    complete windows only
    selected windows may not overlap

These choices remain frozen.

### 4. Action clipping

The publication specifies an L-infinity relative perturbation bound.

Canonical reproduction additionally clips Walker2d actions to [-1, 1].

This remains a frozen reproduction choice.

### 5. Feature preprocessing

Raw state-action decision units are the canonical source setting.

The publication does not operationally specify an additional
state/action scaling procedure before clustering.

Canonical reproduction uses no feature scaling.

### 6. Cluster count

Walker2D canonical cluster count is k=8.

Predeclared k=6 and k=10 sensitivity experiments did not recover a
strong attack effect.

Cluster count alone therefore does not explain the current discrepancy.

## Mechanism-audit questions

The audit will measure:

1. How much deduplication reduces clean distinct pattern count.
2. How often a selected poisoned window actually changes pattern.
3. Conditional on changing pattern, how often frequency increases.
4. How many modified transition cluster assignments actually change.
5. How many targeted rare pattern types disappear completely.
6. How much global occurrence mass of targeted patterns is removed.
7. Whether overall clean-to-poison pattern diversity decreases.
8. How many pattern types disappear and how many new types are created.
9. How many candidate windows are rejected because of overlap.
10. How frequently modified actions end at an action bound.

## Scientific rule

The purpose is diagnosis, not optimization.

A mechanism sensitivity may be proposed only after these canonical
measurements are recorded.

Any such sensitivity must be separately named and must never replace
the canonical Gate-B result retrospectively.

## Deduplication-order source-fidelity diagnostic

A separately frozen clean-data diagnostic compared:

- D0_CANONICAL:
  construct length-5 original-position windows, then remove
  consecutive repeated cluster labels within each window.

- D3_TRAJECTORY_DEDUP_BEFORE_WINDOW:
  remove consecutive repeated cluster labels over each completed
  trajectory first, then construct length-5 windows over the
  compressed label sequence.

Both variants reused the exact same canonical raw-data k=8 KMeans
clustering for each attack seed. No poisoned dataset or learner
performance was used.

Observed three-seed means:

D0_CANONICAL:
- completed-label fraction retained: 100.00%
- raw distinct patterns: 11108.67
- diagnostic distinct patterns: 6281.00
- distinct-pattern reduction: 43.46%

D3_TRAJECTORY_DEDUP_BEFORE_WINDOW:
- completed-label fraction retained: 27.53%
- raw distinct patterns: 11108.67
- diagnostic distinct patterns: 8884.67
- distinct-pattern reduction: 20.02%

D3 removes approximately 72.47% of repeated trajectory labels, but
this must not be conflated with the source paper's reported nearly
80% reduction in the number of distinct decision patterns.

The D3 interpretation therefore does not explain the source/reproduction
deduplication discrepancy.

The publication's method description also more directly supports the
canonical window-then-deduplicate interpretation: a sequence is
extracted first, converted to decision-unit labels, and consecutive
repeated units are then merged to form a decision pattern.

Status:
DEDUP_ORDER_DIAGNOSTIC_NO_EXPLANATION

## Sequence-enumeration source-fidelity diagnostic

A separately frozen clean-data diagnostic compared the canonical
overlapping sequence enumeration against non-overlapping length-5
enumeration.

All conditions reused the same canonical raw-data k=8 KMeans
clustering within each attack seed. Deduplication remained
window-then-deduplicate. No poisoned dataset or learner performance
was used.

Observed three-seed means:

D0_OVERLAPPING_STRIDE_1:
- window count: 995235
- raw distinct patterns: 11108.67
- deduplicated distinct patterns: 6281.00
- distinct-pattern reduction: 43.46%
- dedup-affected window fraction: 94.61%
- average deduplicated pattern length: 2.0935

D4_NONOVERLAPPING_STRIDE_5_OFFSET_0:
- window count: 199800
- raw distinct patterns: 6031.67
- deduplicated distinct patterns: 3347.33
- distinct-pattern reduction: 44.50%
- dedup-affected window fraction: 94.47%
- average deduplicated pattern length: 2.0988

The complete predeclared stride-5 offset sensitivity produced:

- offset 0: 44.50%
- offset 1: 44.74%
- offset 2: 44.60%
- offset 3: 44.79%
- offset 4: 44.73%

The range across all predeclared offsets was only 0.29 percentage
points.

Therefore sequence stride and trajectory-relative offset do not explain
the discrepancy between the canonical reproduction and the source
paper's descriptive nearly-80% distinct-pattern reduction.

No offset is selected as preferred based on these results.

Status:
SEQUENCE_ENUMERATION_DIAGNOSTIC_NO_EXPLANATION

## Deduplication-metric reconciliation

A final clean-data-only diagnostic evaluated multiple possible
deduplication statistics using the exact canonical k=8, raw-data,
length-5, stride-1, window-then-deduplicate pipeline.

No attack variant was introduced and no learner performance was used.

Three-seed means were:

- canonical distinct-type reduction: 43.46%
- window-instance changed fraction: 94.61%
- distinct raw-sequence-type changed fraction: 64.22%
- total label-token reduction: 58.13%
- average deduplicated pattern length: 2.0935
- full-length pattern occurrence fraction: 5.39%
- full-length distinct-pattern fraction: 63.27%
- non-equivalent total-window denominator reduction: 99.37%

The 99.37% quantity compares the number of unique deduplicated pattern
types against the total number of window occurrences and therefore does
not use equivalent pre/post distinct-pattern denominators. It cannot be
interpreted as a reduction in the number of distinct decision patterns.

The 94.61% window-instance quantity similarly measures how often
deduplication changes an individual window, rather than reduction in the
number of distinct pattern types.

None of the predeclared semantically distinct metrics reconciles the
canonical 43.46% distinct-type reduction with the source paper's
descriptive nearly-80% distinct-pattern reduction.

Together with the previously frozen source-fidelity diagnostics:

- literal length-6 interpretation: no explanation;
- per-dimension z-score preprocessing: no explanation;
- trajectory-level deduplication before windowing: no explanation;
- non-overlapping stride-5 enumeration and all offsets: no explanation;
- k=6/k=10 cluster sensitivity: no strong learner degradation recovery.

The source-fidelity discrepancy therefore remains unresolved.

This is not evidence that any alternative metric should replace the
canonical one. No additional attack interpretation will be selected
post hoc solely to reproduce the source statistic.

Status:
SOURCE_FIDELITY_DISCREPANCY_UNRESOLVED

## Reachability-oracle diagnostic

A separately frozen diagnostic tested whether higher-frequency CSDPC
decision patterns are geometrically reachable inside the exact
canonical eta=0.05 state/action perturbation boxes.

The oracle used the frozen raw-data k=8 clustering and the canonical
selected windows. It solved closed KMeans Voronoi-cell feasibility
under the perturbation box constraints and did not train any learner.

The strongest integrity check passed:

- actual_pattern_reachable_fraction = 1.0 for every attack seed and
  both rho=0.01 and rho=0.05.

This confirms that every realized canonical poisoned pattern was
contained in the oracle reachable-pattern set.

For rho=0.01, three-seed means were:

- transitions with more than one reachable cluster: 30.97%
- transitions restricted to their source cluster: 69.03%
- windows for which a pattern change is geometrically possible: 82.73%
- windows for which a higher-frequency pattern is reachable: 72.33%
- canonical 100-candidate frequency-improvement fraction: 54.48%
- canonical attainment of oracle-best frequency: 61.63%
- oracle search-gap fraction: 38.37%
- mean source frequency: 1.31
- mean realized target frequency: 420.62
- mean oracle-best frequency: 1686.23

For rho=0.05, three-seed means were:

- transitions with more than one reachable cluster: 27.69%
- transitions restricted to their source cluster: 72.31%
- windows for which a pattern change is geometrically possible: 78.09%
- windows for which a higher-frequency pattern is reachable: 68.63%
- canonical 100-candidate frequency-improvement fraction: 51.26%
- canonical attainment of oracle-best frequency: 66.67%
- oracle search-gap fraction: 33.33%
- mean source frequency: 10.94
- mean realized target frequency: 785.73
- mean oracle-best frequency: 2341.72

The oracle therefore establishes two simultaneous limitations.

First, eta=0.05 imposes genuine geometric restrictions. Approximately
69--72% of individual selected transitions cannot leave their source
cluster.

Second, geometry does not explain the full weakness of the realized
attack. At the sequence-window level, approximately 69--72% of
selected windows can reach a higher-frequency pattern, whereas the
canonical 100-candidate search realizes such an improvement in only
approximately 51--54% of windows.

The canonical uniform random candidate search therefore leaves a
meaningful fraction of geometrically reachable attack strength unused.

Status:

REACHABILITY_ORACLE_VALIDATED
CANDIDATE_SEARCH_BOTTLENECK_SUPPORTED
GEOMETRY_LIMITATION_ALSO_PRESENT

Canonical Gate B remains INCONCLUSIVE.

## Candidate-search sensitivity

Following the reachability-oracle result, a separately frozen
learner-free sensitivity examined nested uniform candidate pools:

- C100: exact canonical 100-candidate result
- C500: C100 plus 400 deterministic extension candidates
- C1000: C500 plus another 500 extension candidates

The pools were nested, so target-frequency performance could not
decrease solely because different random candidate sets were used.

All integrity checks passed, including exact C100 agreement with the
canonical oracle reference.

At rho=0.01, three-seed means changed from C100 to C1000 as follows:

- frequency-improvement fraction: 54.48% -> 59.88%
- binary oracle opportunity capture: 75.24% -> 82.71%
- mean target-pattern frequency: 420.62 -> 588.91
- selected-source-pattern eradication: 45.48% -> 50.42%
- selected-source occurrence-mass reduction: 47.23% -> 51.89%
- global distinct-pattern reduction: 18.22% -> 20.36%

At rho=0.05:

- frequency-improvement fraction: 51.26% -> 56.48%
- binary oracle opportunity capture: 74.65% -> 82.26%
- mean target-pattern frequency: 785.73 -> 1074.40
- selected-source-pattern eradication: 23.28% -> 26.31%
- selected-source occurrence-mass reduction: 31.95% -> 35.97%
- global distinct-pattern reduction: 19.03% -> 21.75%

The C500-to-C1000 increment was substantially smaller than the
C100-to-C500 increment for both local and global metrics, indicating
diminishing returns from simply increasing uniform candidate count.

Candidate search therefore contributes to the weakness of the
canonical attack, but increasing candidate count alone does not
remove the larger coverage-collapse saturation.

Selection-side quantities were invariant across candidate pools.

At rho=0.01:
- fully selected source-pattern fraction: 86.41%
- selected-source occurrence completion: 89.06%

At rho=0.05:
- fully selected source-pattern fraction: 45.45%
- selected-source occurrence completion: 48.75%

The low completion at rho=0.05 cannot be corrected by candidate
search because candidate generation occurs only after windows have
already been selected.

Status:

CANDIDATE_SEARCH_SENSITIVITY_VALIDATED
CANDIDATE_COUNT_EFFECT_SUPPORTED
C500_TO_C1000_DIMINISHING_RETURNS
CANDIDATE_SEARCH_BOTTLENECK_CONFIRMED_BUT_NOT_SUFFICIENT
SELECTION_COMPLETION_BOTTLENECK_REMAINS
CQL_CANDIDATE_SEARCH_SENSITIVITY_DEFERRED

Canonical Gate B remains INCONCLUSIVE.

## Selection-semantics diagnostic

A separately frozen learner-free diagnostic compared:

- S0: canonical occurrence-level selection with non-overlap;
- S1: the same rare-pattern occurrence ordering with overlap allowed
  and a unique-transition footprint budget;
- S2: rare pattern types selected atomically as a strict prefix,
  with all occurrences included and overlap allowed.

At rho=0.01, three-seed means were:

S0:
- selected windows: 2000
- selected source-pattern types: 1808
- fully selected source-pattern fraction: 86.41%
- source-occurrence completion: 89.06%

S1:
- selected windows: 2965
- selected source-pattern types: 2617
- fully selected source-pattern fraction: 99.92%
- source-occurrence completion: 99.93%
- overlap reuse fraction: 32.55%
- overlapped unique-transition fraction: 33.65%

S2:
- selected windows: 2963
- selected source-pattern types: 2615
- fully selected source-pattern fraction: 100%
- source-occurrence completion: 100%
- budget utilization: 99.98%

At rho=0.05:

S0:
- selected windows: 10000
- selected source-pattern types: 3824.33
- fully selected source-pattern fraction: 45.45%
- source-occurrence completion: 48.75%

S1:
- selected windows: 18550.33
- selected source-pattern types: 5105.67
- fully selected source-pattern fraction: 99.97%
- source-occurrence completion: 99.92%
- overlap reuse fraction: 46.08%
- overlapped unique-transition fraction: 50.54%

S2:
- selected windows: 18539.33
- selected source-pattern types: 5104.33
- fully selected source-pattern fraction: 100%
- source-occurrence completion: 100%
- budget utilization: 99.93%

S1 and S2 are nearly identical structurally. Pattern-type atomicity
therefore contributes little beyond permitting overlapping occurrences.

The major difference from the canonical selector is the non-overlap
constraint. Removing it approximately restores complete targeting of
rare source-pattern occurrences and prevents the selector from moving
as quickly into higher-frequency source patterns.

However, the resulting overlap burden is substantial. At rho=0.05,
approximately half of the unique selected transitions participate in
more than one selected window, with transition multiplicity reaching 5.

Independent per-window perturbations therefore cannot yet be treated
as a uniquely defined poisoned dataset, because overlapping windows
may prescribe incompatible modifications to the same transition.

Status:

SELECTION_SEMANTICS_DIAGNOSTIC_VALIDATED
NONOVERLAP_CONSTRAINT_MAJOR_STRUCTURAL_BOTTLENECK
PATTERN_TYPE_ATOMICITY_MINIMAL_INCREMENTAL_EFFECT
OVERLAP_CONFLICT_RESOLUTION_REQUIRED
CQL_DEFERRED

Canonical Gate B remains INCONCLUSIVE.

## Overlap target-conflict diagnostic

A separately frozen learner-free diagnostic independently generated
canonical C100 perturbation proposals for all windows selected under
the S2 type-complete overlapping interpretation and measured whether
overlapping windows requested incompatible KMeans target labels.

At rho=0.01:

- 33.61% of unique selected transitions participated in overlap;
- 67.09% of selected windows touched at least one overlap;
- 8.66% of overlapped transitions had incompatible target labels;
- conflicting transitions represented only 2.91% of the complete
  unique transition footprint;
- 5.20% of proposal slots participated in a conflict;
- pairwise label disagreement on overlaps was 7.55%;
- 21.07% of selected windows touched at least one target-label conflict;
- mean number of distinct proposed labels per overlapped transition
  was 1.087.

At rho=0.05:

- 50.56% of unique selected transitions participated in overlap;
- 85.06% of selected windows touched at least one overlap;
- 7.60% of overlapped transitions had incompatible target labels;
- conflicting transitions represented only 3.84% of the complete
  unique transition footprint;
- 6.25% of proposal slots participated in a conflict;
- pairwise label disagreement on overlaps was 6.22%;
- 24.76% of selected windows touched at least one target-label conflict;
- mean number of distinct proposed labels per overlapped transition
  was 1.077.

Thus overlap itself is widespread, especially at rho=0.05, but most
overlapping independent C100 proposals agree on the required KMeans
target label.

A realizable type-complete overlapping attack is therefore plausible,
but an explicit conflict-resolution convention is still required.
Because roughly one quarter of selected windows touch at least one
conflicting transition at rho=0.05, the merge rule should be tested
before any learner is trained.

Status:

OVERLAP_TARGET_CONFLICT_DIAGNOSTIC_VALIDATED
OVERLAP_WIDESPREAD
TARGET_LABEL_CONFLICT_SPARSE_PER_TRANSITION
WINDOW_LEVEL_CONFLICT_EXPOSURE_NONTRIVIAL
TYPE_COMPLETE_OVERLAP_REALIZATION_FEASIBLE
CONFLICT_RESOLUTION_SENSITIVITY_REQUIRED
CQL_DEFERRED

Canonical Gate B remains INCONCLUSIVE.
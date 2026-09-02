# CSDPC

Independent reproduction of:

Collapsing Sequence-Level Data-Policy Coverage via Poisoning Attack
in Offline Reinforcement Learning.

Group-2 implementation order:

clean dataset
-> raw state-action decision units
-> k-means clustering
-> episode-safe sequence windows
-> consecutive-label deduplication
-> decision-pattern frequencies
-> rare-pattern selection
-> CSDPC perturbation
-> immutable poisoned dataset + metadata

Source-under-specified components must be documented before a
reproduction convention is chosen.

CSDPC must not contain learner-specific DT, CQL, or BC logic.
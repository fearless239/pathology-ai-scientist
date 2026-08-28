# Comparison training policy repair

The 2026-08-28 failure was a completed method fit capped at 6 epochs compared
with a baseline capped at 12 epochs. Three 8-epoch search candidates exhausted
the 30-epoch node budget. Equal stopping rules do not make unequal caps a fair
comparison. Historical weights, manifests, metrics and approved contracts must
not be rewritten to hide the mismatch.

The stage-3 manager now binds its inherited baseline to verified immutable
evidence and records its explicit training policy. Generation and debug prompts
receive that policy. They must reserve the final cap before assigning search
epochs and declare a literal `FINAL_TRAINING_PLAN`. Preflight rejects missing,
mismatched or over-budget plans and contradictory ordinary final-cap constants.
The post-execution manifest is independently compared with the same policy;
different actual epoch counts remain allowed within the policy.

This is not arbitrary Python control-flow verification. A declaration is not
proof that every runtime branch follows it; output checks remain necessary.
Existing inference, resource and scientific acceptance guards are retained.

Tests cover 12+[5,5,5], rejected 6+[8,8,8], rejected 12+[8,8,8], actual epoch
differences, contradictory final-cap constants, and rejection through the real
interpreter before the external runner. The full suite passed with project-local
temporary files, followed by the expanded focused tests. No paid validation ran.

`scripts/prepare-comparison-recovery.py` explicitly backs up a stopped task,
checks verified evidence and the exact fairness failure, excludes its invalid
stage-3 result from success eligibility without deleting artifacts, preserves
stage-1/2 journals, and reopens stage 3 with three new attempts. It refuses tasks
already frozen/tested or with other acceptance failures. It does not launch work.
The prepared pathtest-001 budget, contract files and evidence were hash-checked
unchanged; only checkpoint eligibility and UI stage projection were updated.

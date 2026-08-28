# PDF-only recovery (2026-08-28)

The task reached `translation_completed` after training, freezing and sealed
testing. Its English PDF failed on literal Unicode alpha. An external full-LaTeX
repair failed numeric-content validation, leaving phase `repairing`; subsequent
restarts stopped before attempting compilation.

The exporter now maps common Greek and mathematical Unicode symbols to LaTeX
commands in both prose and existing inline equations. No manuscript numbers,
claims, references or evidence are rewritten by this conversion.

Compilation recovery first attempts local compilation of the retained source
(or deterministically regenerated source). An unresolved/rejected external
repair still blocks further external requests if local compilation fails.
Consumed repair counts survive this path. A successful local compile records
`compiled`. Newly rejected numeric edits are saved as rejected responses with
phase `rejected`, rather than being misrepresented as unfinished requests.
The numeric-content guard has not been weakened or removed.

Regression tests cover both unchanged and regenerated source recovery, no
request resend, durable rejected responses, repair caps, and Unicode equations.
The actual English and Chinese manuscript copies compiled twice with the existing
Docker image and passed the existing citation/overfull-box quality gate. These
are compilation checks, not a new scientific/content review or full visual QA.

No task state was reset, no historical response was deleted, and no live LLM,
training or sealed-test call was made for verification. A manual frontend restart
can continue from the existing `translation_completed` state into PDF/archive.

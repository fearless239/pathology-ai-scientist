# Paper Model Quality Evaluation

Compare models only after freezing an identical `paper_evidence.json`, verified literature set, prompt,
token ceiling, and LaTeX/PDF quality gate. Evaluate blinded outputs on factual consistency, citation
coverage, method completeness, statistical restraint, review-resolution rate, readability, compilation
quality, token usage, cost, and latency.

Use the current writer as the baseline. First rerun it with the corrected evidence pipeline; then compare a
stronger model on the exact same inputs. Change the default only when the stronger model improves the
predeclared quality score across at least three representative tasks without introducing unsupported claims
and with an acceptable cost increase. A model comparison must not reuse test results for experiment design.

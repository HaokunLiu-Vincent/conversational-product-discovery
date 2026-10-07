# Evaluation

Run the complete development preference-extraction and state-tracking evaluation with:

```bash
python -m product_discovery.evaluation.run_development --output-dir evaluation/results/dev
```

The runner accepts only records marked `split=development`, does not load the product
catalogue, and never performs retrieval or response generation. It reports the original
strict metrics alongside exact core-state metrics (hard fields plus `usage`) and executable
metrics (hard fields only). All comparisons use production canonicalization and exact values;
there is no fuzzy or semantic scoring.

Outputs under the selected directory include reusable raw predictions, JSON/CSV metrics,
error records, a Markdown summary, deterministic error taxonomy, and cascading state-drift
analysis. Generated results are ignored by Git.

The benchmark does not contain sufficient reference responses or supported product claims
for an automatic response-faithfulness score. Response evaluation requires a separate,
explicitly supported procedure.

Clarification questions are scored by exact string equality. Semantically similar wording
receives no additional credit.

Do not inspect or execute the protected held-out split from this runner.

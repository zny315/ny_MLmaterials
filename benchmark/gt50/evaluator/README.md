# GT50 Hierarchical Evaluator v18

Use the repository wrapper:

```bash
python scripts/replay_gt50.py --output-dir outputs/gt50_replay
```

Or invoke directly:

```bash
python benchmark/gt50/evaluator/evaluate_gt50.py \
  --gt benchmark/gt50/ground_truth/GT50_EVA_hierarchical_ground_truth_publication_clean.xlsx \
  --direct-dir benchmark/gt50/extraction_outputs/direct_llm \
  --expert-dir benchmark/gt50/extraction_outputs/expert_prompt_only \
  --split-dir benchmark/gt50/extraction_outputs/task_decomposition_only \
  --full-dir benchmark/gt50/extraction_outputs/full_pipeline \
  --output-dir outputs/gt50_replay
```

Task Decomposition Only already includes Articles 13/544 reruns. Full Pipeline
already includes restored Article 19 labels. Test rules with
`python benchmark/gt50/evaluator/test_evaluator.py`.

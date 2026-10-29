# data/synthetic
Track A: the fictional company **Quillfen Electronics**.

- `kb/`: 51 markdown docs (front matter + `##` sections; each section is one chunk `doc_id#section-slug`).
- `scenarios.yaml`: source of truth for all gold labels (hand-written).
- `known_issues.jsonl`: 20 known issues, built from the `ki-*` bulletins by `scripts/generate_synthetic.py`.
- `tickets/`: generated tickets with gold labels, split 60/20/20 by scenario. `test.jsonl` is frozen (`test.sha256`); never use it in prompts, the known-issues list, or tuning.

To change a label, edit `scenarios.yaml` and rerun the generator (cached LLM calls make reruns fast).

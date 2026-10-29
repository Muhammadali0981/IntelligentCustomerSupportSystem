# Changelog

One dated line per meaningful change.

- 2026-10-28: Phase 1. Project scaffold, configs (models, thresholds, datasets, home/lab profiles), global Python 3.11.8 (no venv; missing packages added without changing existing ones), Ollama installed, home models pulled, scripts/check_setup.py.
- 2026-10-28: Measured VRAM per num_ctx (4B: 8192 fast / 16384 reasoning; 9B: 8192 with ~12% CPU offload); recorded in home.yaml and README.
- 2026-10-28: USE_TF=0 set in src/config.py to avoid a global TensorFlow/Keras 3 import error in transformers. check_setup.py: 8/8 pass.
- 2026-10-29: Phase 2. download_data.py (all 5 core + 3 optional datasets incl. banking77 via parquet branch), prepare_data.py (label maps, GPU embedding dedup at cosine 0.95, stratified 70/15/15 split), Tobi-Bueck spot-check CSV.
- 2026-10-29: Bitext real category names differ from brief (CANCEL, SHIPPING); mapping updated in configs/datasets.yaml. Tobi-Bueck file fixed to aa_dataset-tickets-multi-lang-5-2-50-version.csv.
- 2026-10-29: Track A "Quillfen Electronics": 51 KB docs (195 chunks), 20 known issues, scenarios.yaml with all gold labels, generate_synthetic.py (LLM writes text only, scenario-level split, frozen test hash).
- 2026-10-29: src/llm (Ollama client with disk cache + Pydantic retry), src/retrieval/chunking.py (section chunk IDs), src/retrieval/embedding.py, label_wixqa.py, data/DATASETS.md, tests/test_data_integrity.py.

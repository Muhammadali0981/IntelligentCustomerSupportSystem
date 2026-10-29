# Intelligent Customer Support System

University AI course project. A customer support ticket is:

1. **classified** (billing, account_access, technical, shipping, other),
2. **routed** as KNOWN (matches a known issue) or NEW,
3. **answered** with basic RAG (KNOWN, thinking off) or reasoning RAG (NEW, same model with thinking on, wider retrieval + reranking),
4. **checked** for grounding (citations must come from retrieved chunks); otherwise escalated to a human,
5. **logged** (route, retrieved IDs, tokens, per-stage latency, cost) for comparison against simpler baselines.

Everything runs locally with Ollama. An optional hosted API arm can be enabled with a key in `.env`, but nothing requires it.

## Status

| Phase | Content | Status |
|---|---|---|
| 1 | Setup, configs, `check_setup.py` | done |
| 2 | Data (public + synthetic company) | done (pending human review CSVs) |
| 3 | Classification | - |
| 4 | Retrieval | - |
| 5 | Routing | - |
| 6 | Answering + guard | - |
| 7 | Evaluation | - |
| 8 | Demo + docs | - |

## Install

Tested on Windows 11, Python 3.11.8, NVIDIA RTX 3060 Ti (8 GB). The home machine uses its **global** Python 3.11 environment (no venv).

1. **Python packages.** Install torch with CUDA first, then the rest:
   ```
   py -3.11 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   py -3.11 -m pip install -r requirements.txt
   ```
   To add only missing packages to an existing environment without upgrading anything, run `pip freeze > constraints.txt` first and then install with `-c constraints.txt`.
2. **Ollama** (official installer, no admin needed): `winget install Ollama.Ollama`. Set `OLLAMA_MAX_LOADED_MODELS=1` as a user environment variable before starting it.
3. **Models:**
   ```
   ollama pull qwen3.5:4b-q4_K_M
   ollama pull qwen3.5:9b-q4_K_M      # optional larger reasoning model
   ```
   The embedder and rerankers download into `models/cache/` on first use.
4. **Optional API arm:** copy `.env.example` to `.env` and fill in a key.
5. **Verify:** `py -3.11 scripts/check_setup.py`

### Environment notes

- `src/config.py` sets `USE_TF=0`. The home machine also has TensorFlow with Keras 3 installed globally, which makes `transformers` 4.57 fail on import; we only use PyTorch.
- On Windows without Developer Mode, the Hugging Face cache copies files instead of symlinking them. This works but uses more disk space.

## Running each phase

```
py -3.11 scripts/check_setup.py          # Phase 1: verify environment
py -3.11 scripts/download_data.py --optional   # Phase 2: public data -> data/raw/ (re-runnable)
py -3.11 scripts/prepare_data.py         # Phase 2: map, dedup, split -> data/processed/, review CSV
py -3.11 scripts/generate_synthetic.py   # Phase 2: Track A known issues + tickets (LLM-cached)
py -3.11 scripts/label_wixqa.py          # Phase 2: LLM labels for 400 WixQA questions -> review CSV
py -3.11 -m pytest tests -q              # leakage / integrity checks
```

Human review files (fill in the marked columns): `data/review/tobi_bueck_label_spotcheck.csv`, `data/review/synthetic_tickets_review.csv`, `data/review/wixqa_category_labels.csv`.

## Configuration

- `configs/models.yaml`: model tags (verified against the Ollama registry), quantization, API prices (marked "verify before reporting").
- `configs/profiles/home.yaml` / `lab.yaml`: everything hardware-dependent (which model per route, `num_ctx`, devices). Select with `SUPPORT_PROFILE=lab` or `--profile lab`.
- `configs/thresholds.yaml`: routing/escalation thresholds (tuned on dev splits only).
- `configs/datasets.yaml`: dataset IDs, label mappings, sample sizes, seed.

### Ollama settings

| Setting | Value | Why |
|---|---|---|
| `OLLAMA_MAX_LOADED_MODELS` | 1 (home) | 8 GB VRAM holds one LLM |
| `keep_alive` | `-1` during evaluation, `5m` otherwise | no reload cost inside timed runs |
| `num_ctx` fast route | 8192 | Ollama's VRAM-based default is 4096, too small for RAG |
| `num_ctx` reasoning route | 16384 | room for 8 chunks + thinking tokens |
| `temperature` | 0 for classification, routing, judging | reproducibility |

### Context lengths (measured 2026-10-28, `ollama ps`)

| Model | num_ctx | VRAM | On GPU |
|---|---|---|---|
| qwen3.5:4b-q4_K_M | 8192 | 3.13 GiB | 100% |
| qwen3.5:4b-q4_K_M | 16384 | 3.40 GiB | 100% |
| qwen3.5:4b-q4_K_M | 32768 | 3.83 GiB | 100% |
| qwen3.5:9b-q4_K_M | 4096 | 5.22 GiB | 100% |
| qwen3.5:9b-q4_K_M | 8192 | 5.95 GiB | 88% (partial CPU offload) |

Other apps used 0.8-3.6 GB of VRAM during these measurements. Close games and video players before timing runs.

## Data licences

Public data is never committed; the scripts rebuild it. Note:
- **Tobi-Bueck/customer-support-tickets** is CC BY-NC 4.0: non-commercial use only.
- **Bitext** is CDLA-Sharing-1.0: derivatives must be shared under the same licence.

Full sources, licences and citations: `data/DATASETS.md`.

Track A uses **Quillfen Electronics**, an invented company; any resemblance to a real business is unintended.

## Layout

See the README in each folder. In short: `configs/` settings, `data/` datasets, `src/` the system, `eval/` evaluation, `scripts/` setup and data scripts, `app/` Streamlit demo, `results/` outputs, `tests/` unit tests.

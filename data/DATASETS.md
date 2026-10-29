# Datasets

All IDs, files, licences and column names below were checked against the Hugging Face API and the downloaded files on 2026-10-28/29. Raw and processed public data are **not** committed; `scripts/download_data.py` and `scripts/prepare_data.py` rebuild them (seed 42, `configs/datasets.yaml`). Row counts come from `scripts/prepare_data.py` output; see also `results/tables/data_summary.md`.

**Licence notes:** Tobi-Bueck is **CC BY-NC 4.0 (non-commercial only)**. Bitext is **CDLA-Sharing-1.0**: anything derived from it must be shared under the same licence.

## Track B: public datasets

### 1. Bitext customer support (classification)

| | |
|---|---|
| ID | `bitext/Bitext-customer-support-llm-chatbot-training-dataset`, single `train` split |
| File | `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv` |
| Licence | CDLA-Sharing-1.0 |
| Columns | `flags`, `instruction`, `category`, `intent`, `response` (only `instruction` is used as input) |
| Raw rows | 26,872 (27 intents, 11 categories; no technical category) |
| After dedup | 10,342 (cosine > 0.95 with Qwen3-Embedding-0.6B, before splitting) |
| Used | 150 per intent (cancel_order: 119, all that survived dedup) = 4,019 rows |

Mapping (intent rules override category rules). The real category names differ from the project brief: `CANCEL` (not CANCELLATION_FEE), `SHIPPING` (not SHIPPING_ADDRESS); `SUBSCRIPTION` holds only `newsletter_subscription`.

| Our label | Bitext categories / intents |
|---|---|
| billing | INVOICE, PAYMENT, REFUND, CANCEL (check_cancellation_fee) |
| account_access | intents create_account, delete_account, edit_account, recover_password, registration_problems, switch_account (= all of ACCOUNT) |
| shipping | DELIVERY, SHIPPING (change/set_up_shipping_address), intent track_order |
| other | FEEDBACK, CONTACT, SUBSCRIPTION, intents place_order, cancel_order, change_order |

Citation: Bitext Innovations, *Bitext Customer Support LLM Chatbot Training Dataset*, Hugging Face, https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

### 2. Tobi-Bueck customer support tickets (adds technical)

| | |
|---|---|
| ID | `Tobi-Bueck/customer-support-tickets` |
| File used | `aa_dataset-tickets-multi-lang-5-2-50-version.csv` **only** (the newest; the three files overlap) |
| Licence | CC BY-NC 4.0, synthetic |
| Columns | `subject`, `body`, `answer`, `type`, `queue`, `priority`, `language`, `version`, `tag_1`...`tag_8` |
| Raw rows | 28,587 (16,338 English); 12,011 English rows in mapped queues |
| After dedup | 7,945 |
| Used | 1,000 technical + 500 billing = 1,500 |

Input = `subject` + `body` only (never `answer` or tags). Queue labels are noisy: 100 rows are exported to `data/review/tobi_bueck_label_spotcheck.csv` for a human spot-check.

| Our label | Queues |
|---|---|
| technical | Technical Support, IT Support, Product Support, Service Outages and Maintenance |
| billing | Billing and Payments |
| (dropped) | Customer Service, Returns and Exchanges, Sales and Pre-Sales, Human Resources, General Inquiry |

Citation: T. Bueck, *customer-support-tickets*, Hugging Face, https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets

**Classification set** (Bitext + Tobi-Bueck combined, stratified 70/15/15 by source x label): train 3,863 / dev 827 / test 829. Billing comes from both sources so we can test whether "technical" is learned or just Tobi-Bueck's email style.

### 3. WixQA (retrieval, answer quality, real tickets)

| | |
|---|---|
| ID | `Wix/WixQA`, configs `wix_kb_corpus`, `wixqa_expertwritten`, `wixqa_simulated`, `wixqa_synthetic` (each a single split named `train`) |
| Licence | MIT |
| KB columns | `id`, `url`, `contents`, `title`, `html_content`, `article_type` (`title` and `html_content` are extra vs. the brief) |
| QA columns | `question`, `answer`, `article_ids` |
| KB | 6,221 articles: 4,109 article, 2,049 feature_request, **63 known_issue** |
| Test | expertwritten 200 + simulated 200 = 400 real questions (gold article types: 478 article, 10 feature_request, 0 known_issue) |
| Dev | wixqa_synthetic 6,221 (tuning only, never test evidence) |

The 400 real questions are labelled with our 5 categories by `qwen3.5:9b` (not the 4B model under test) and exported to `data/review/wixqa_category_labels.csv` for human verification.

Citation: D. Cohen, L. Burg, S. Pykhnivskyi, H. Gur, S. Kovynov, O. Atzmon, G. Barkan. *WixQA: A Multi-Dataset Benchmark for Enterprise Retrieval-Augmented Generation.* arXiv:2505.08643, 2025.

### 4. CLINC150 (off-domain rejection)

| | |
|---|---|
| ID | `clinc/clinc_oos`, config `plus` |
| Licence | CC BY 3.0 |
| Columns | `text`, `intent` (int; names saved to `data/raw/clinc/plus__labels.json`; `oos` = 42) |
| Used | validation 3,100 (100 oos) for thresholds; test 5,500 (1,000 oos) for evaluation |

Reported as "off-domain rejection", separate from new-in-domain routing.

Citation: S. Larson et al. *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction.* EMNLP-IJCNLP 2019.

### 5. RAGBench (judge validation)

| | |
|---|---|
| ID | `galileo-ai/ragbench`, configs `techqa`, `emanual` |
| Licence | CC BY 4.0 |
| Used columns | `question`, `documents`, `response`, `documents_sentences`, `response_sentences`, `sentence_support_information`, `unsupported_response_sentence_keys`, `adherence_score` |
| Rows | techqa 1,192 / 304 / 314; emanual 1,054 / 132 / 132 (train / validation / test) |

Citation: R. Friel, M. Belyi, A. Sanyal. *RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems.* arXiv:2407.11005, 2024. (Unverified: the dataset card has no citation block; check before reporting.)

### Optional

| Dataset | ID | Licence | Status |
|---|---|---|---|
| TechQA-RAG-Eval | `nvidia/TechQA-RAG-Eval` | Apache-2.0 | `train.json` downloaded (910 rows; `is_impossible` is a string "True"/"False"; `contexts` inline) |
| banking77 | `PolyAI/banking77`, `revision="refs/convert/parquet"` | CC BY 4.0 | loads fine: train 10,003, test 3,080 |
| MakTek FAQs | `MakTek/Customer_support_faqs_dataset` | Apache-2.0 | 200 FAQs, phrasing ideas only |

banking77 citation: I. Casanueva et al. *Efficient Intent Detection with Dual Sentence Encoders.* NLP4ConvAI @ ACL 2020.

Not used (as instructed): `suraj520/customer-support-ticket-dataset`, doc2dial / MultiDoc2Dial, Twitter customer support.

## Track A: synthetic company (Quillfen Electronics)

Quillfen Electronics is entirely fictional (online electronics retailer, stores in US/Canada/UK/EU, "Quillfen Plus" membership). All content was written for this project; it is committed to git.

| Item | Count | File |
|---|---|---|
| KB documents | 51 (12 policies, 7 FAQs, 12 troubleshooting, 20 known-issue bulletins) | `data/synthetic/kb/*.md` |
| KB chunks (one per `##` section) | 195 | IDs `doc_id#section-slug` |
| Known issues | 20 | `data/synthetic/known_issues.jsonl` |
| Scenarios (gold labels) | 77 | `data/synthetic/scenarios.yaml` |
| Tickets | see `results/tables/synthetic_summary.md` | `data/synthetic/tickets/{train,dev,test}.jsonl` |

Deliberate traps for the reasoning route: EU/UK 14-day withdrawal vs 30-day policy; hygiene exclusion for in-ear products; 15% restocking fee only after 14 days; holiday return window; Plus exceptions; Canada-specific membership rules; and an **outdated 2025 shipping FAQ** (`faq-shipping-legacy`, free shipping over $50) that conflicts with the September 2026 policy ($75).

Ticket text is written by `qwen3.5:4b` (temperature 0.9, seeded, cached) from the scenario briefs; all labels come from `scenarios.yaml`. Scenarios (not tickets) are split 60/20/20, stratified by type, so paraphrases never cross splits. The test split is frozen by `tickets/test.sha256`. 50 tickets are exported to `data/review/synthetic_tickets_review.csv` for the team to check.

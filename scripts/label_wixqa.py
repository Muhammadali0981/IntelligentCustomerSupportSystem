"""LLM-label the 400 real WixQA questions (expertwritten + simulated) with our
5 categories, then export them for human verification.

Run:  py -3.11 scripts/label_wixqa.py
Uses the larger local model (route "reasoning_large", thinking off, temperature 0),
NOT the fast model that Phase 3 evaluates, so the classifier is not graded on its
own labels. Humans fill the `human_label` column; Phase 3 uses only verified labels.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_config  # noqa: E402
from src.llm.ollama_client import OllamaLLM  # noqa: E402

PROCESSED = ROOT / "data" / "processed" / "wixqa"
REVIEW = ROOT / "data" / "review"

LABEL_GUIDE = """You label customer support questions sent to Wix, a website-building platform.
Choose exactly one category:
- billing: paying Wix or getting paid through Wix: charges, refunds, invoices, premium plan prices,
  renewals, upgrades or downgrades of paid plans, payment methods, taxes, Wix Payments and payouts.
- account_access: signing in, passwords, two-step verification, account email, account security,
  transferring site ownership, collaborators and roles, closing or deleting an account.
- technical: how to build, edit or configure a site or a Wix feature, domains, mailboxes, SEO, apps,
  integrations, errors and things not working as expected.
- shipping: shipping and delivery of physical orders from a Wix store: shipping rules and rates,
  delivery regions, fulfilment, tracking numbers.
- other: anything else: feedback, feature requests, legal or policy questions, general questions.
If a question fits two categories, pick the one the customer most needs help with."""


class WixLabel(BaseModel):
    category: Literal["billing", "account_access", "technical", "shipping", "other"]
    confidence: float = Field(ge=0.0, le=1.0)


def label_question(llm: OllamaLLM, question: str) -> WixLabel:
    messages = [{"role": "system", "content": LABEL_GUIDE},
                {"role": "user", "content": f"Question:\n{question}\n\nReply as JSON with category and confidence."}]
    label, _ = llm.structured(messages, WixLabel, temperature=0.0, think=False)
    return label


def main() -> None:
    cfg = load_config()
    llm = OllamaLLM(cfg, "reasoning_large")
    df = pd.read_parquet(PROCESSED / "test_real.parquet")
    labels = []
    for i, q in enumerate(df["question"]):
        labels.append(label_question(llm, q))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)}", flush=True)
    df["llm_label"] = [x.category for x in labels]
    df["llm_confidence"] = [x.confidence for x in labels]
    df["llm_model"] = llm.tag
    df.drop(columns=["answer"]).to_parquet(PROCESSED / "test_real_llm_labels.parquet", index=False)

    out = df[["qid", "subset", "question", "llm_label", "llm_confidence"]].copy()
    out["human_label"] = ""
    out["notes"] = ""
    REVIEW.mkdir(parents=True, exist_ok=True)
    out.to_csv(REVIEW / "wixqa_category_labels.csv", index=False, encoding="utf-8-sig")
    print(f"wrote data/review/wixqa_category_labels.csv ({len(out)} rows, model {llm.tag})")
    print(out["llm_label"].value_counts().to_string())


if __name__ == "__main__":
    main()

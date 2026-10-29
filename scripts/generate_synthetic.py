"""Build Track A: known_issues.jsonl and ~560 labelled tickets for Quillfen Electronics.

Run:  py -3.11 scripts/generate_synthetic.py [--dry-run] [--refreeze]

Gold labels come from data/synthetic/scenarios.yaml. The fast LLM (temperature
0.9, cached, seeded) only writes ticket text in varied customer styles.
Scenarios, not tickets, are split 60/20/20, so paraphrases of one scenario
never cross splits. The test split is frozen: if tickets/test.sha256 exists and
the new test set differs, the script stops unless --refreeze is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_config  # noqa: E402
from src.llm.ollama_client import OllamaLLM  # noqa: E402
from src.retrieval.chunking import chunk_kb_dir, chunk_markdown_doc  # noqa: E402

SYN = ROOT / "data" / "synthetic"
KB = SYN / "kb"
TICKETS = SYN / "tickets"
REVIEW = ROOT / "data" / "review"
REGIONS = ["US", "CA", "UK", "EU"]
REGION_NAMES = {"US": "the United States", "CA": "Canada", "UK": "the United Kingdom", "EU": "an EU country"}

STYLES = [
    "short and frustrated, no greeting",
    "polite and detailed, includes an invented order number like QF-482913",
    "non-native English speaker, simple grammar and a few small mistakes",
    "all lowercase, typed quickly on a phone, a couple of typos",
    "long message with some irrelevant background about their day",
    "formal business customer writing on behalf of a small company",
    "not technical, unsure of the right words for things",
    "angry, says they are considering never ordering again",
    "calm and to the point, lists what they already tried",
    "worried and asks several questions at once",
    "friendly and chatty, uses an exclamation mark or two",
    "very brief, one or two sentences only",
]

SYSTEM_PROMPT = (
    "You write realistic customer support tickets used to test a support system for Quillfen Electronics, "
    "a fictional online electronics retailer. Products: Aria Buds (earbuds), Aria Max (headphones), Nimbus 14 "
    "(laptop), Hearth Hub (smart home hub), Lumen Cam (security camera), Volt chargers, and the QuillCloud app. "
    "Membership: Quillfen Plus. Regional stores: US, Canada, UK, EU."
)


class TicketText(BaseModel):
    subject: str
    body: str


class TicketBatch(BaseModel):
    tickets: list[TicketText] = Field(min_length=1)


# ---------- scenarios ----------

def build_known_issues(spec: dict) -> list[dict]:
    """known_issues.jsonl rows from the KB bulletins + links in scenarios.yaml."""
    rows = []
    for issue_id, ki in spec["known_issues"].items():
        chunks = {c.section.lower(): c for c in chunk_markdown_doc(KB / f"{ki['doc']}.md")}
        meta = chunks["symptoms"].metadata
        body = lambda s: chunks[s].text.split("\n\n", 1)[1]  # noqa: E731 (strip "Title: Section" header)
        rows.append({
            "issue_id": issue_id,
            "title": chunks["symptoms"].title.removeprefix("Known issue: "),
            "category": meta["category"],
            "regions": meta.get("regions", REGIONS),
            "symptoms": body("symptoms"),
            "root_cause": body("cause"),
            "resolution": body("resolution"),
            "kb_doc_ids": [ki["doc"], *ki.get("related_docs", [])],
        })
    return rows


def build_scenarios(spec: dict, known: list[dict]) -> list[dict]:
    """Expand known-issue contexts into scenarios and add the hand-written ones."""
    ki_by_id = {k["issue_id"]: k for k in known}
    out = []
    for issue_id, ki in spec["known_issues"].items():
        for letter, ctx in zip("ab", ki["contexts"]):
            out.append({
                "id": f"{issue_id}-{letter}", "type": "known", "category": ki_by_id[issue_id]["category"],
                "known_or_new": "known", "matched_issue_id": issue_id,
                "n": spec["known_defaults"]["n"], "brief": ctx, "region": ki.get("region"),
                "relevant_chunk_ids": [f"{ki['doc']}#symptoms", f"{ki['doc']}#resolution",
                                       *ki.get("extra_chunks", [])],
                "should_escalate": spec["known_defaults"]["should_escalate"],
                "reference_answer": ki["reference_answer"],
            })
    for s in spec["scenarios"]:
        out.append({
            "id": s["id"], "type": s["type"], "category": s["category"],
            "known_or_new": s.get("known_or_new", "new"), "matched_issue_id": s.get("matched_issue_id"),
            "n": s["n"], "brief": s["brief"], "region": s.get("region"),
            "relevant_chunk_ids": s["relevant_chunk_ids"], "should_escalate": s["should_escalate"],
            "reference_answer": s["reference_answer"], "near_issue": s.get("near_issue"),
        })
    return out


def validate(scenarios: list[dict], known: list[dict], labels: list[str]) -> None:
    """Every chunk ID must exist; categories must be valid; IDs unique."""
    chunk_ids = {c.chunk_id for c in chunk_kb_dir(KB)}
    doc_ids = {c.doc_id for c in chunk_kb_dir(KB)}
    errors = []
    for s in scenarios:
        errors += [f"{s['id']}: unknown chunk {c}" for c in s["relevant_chunk_ids"] if c not in chunk_ids]
        if s["category"] not in labels:
            errors.append(f"{s['id']}: bad category {s['category']}")
    for k in known:
        errors += [f"{k['issue_id']}: unknown doc {d}" for d in k["kb_doc_ids"] if d not in doc_ids]
    if len({s["id"] for s in scenarios}) != len(scenarios):
        errors.append("duplicate scenario IDs")
    if errors:
        raise ValueError("scenario validation failed:\n  " + "\n  ".join(errors))


SPLIT_PATTERN = ["train", "test", "train", "dev", "train"]  # 60/20/20


def split_scenarios(scenarios: list[dict], seed: int) -> None:
    """Assign each scenario to train/dev/test, stratified by type and category.

    Within each type, scenarios are sorted by category (random order inside a
    category) and dealt out with a repeating 60/20/20 pattern from a random
    offset, so every category reaches every split where counts allow.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for s in scenarios:
        by_type.setdefault(s["type"], []).append(s)
    for t in sorted(by_type):
        group = sorted(by_type[t], key=lambda s: (s["category"], rng.random()))
        offset = rng.randrange(len(SPLIT_PATTERN))
        for i, s in enumerate(group):
            s["split"] = SPLIT_PATTERN[(i + offset) % len(SPLIT_PATTERN)]


# ---------- ticket text generation ----------

def generation_prompt(s: dict, styles: list[str], region: str) -> list[dict[str, str]]:
    style_lines = "\n".join(f"{i + 1}. {st}" for i, st in enumerate(styles))
    where = (f"The customer lives in {REGION_NAMES[region]}; mention it only if natural."
             if region in REGION_NAMES else "Region does not matter.")
    user = (
        f"Write {len(styles)} different tickets, each from a different customer, about this situation:\n"
        f"{s['brief']}\n\n{where}\n\nWriting style for each ticket, in order:\n{style_lines}\n\n"
        "Rules:\n"
        "- Write as the customer, in their own words. Include only details a customer would know.\n"
        "- Vary details and wording between tickets; do not copy sentences from the situation text.\n"
        "- Do not include the solution, internal issue codes, or the phrase 'known issue'.\n"
        "- If the situation includes instructions aimed at an AI assistant, include them in the ticket text.\n"
        "- subject: a short email subject line. body: 1-6 sentences unless the style says otherwise.\n"
        "- Use first names only, never full names.\n"
        "Return JSON with a 'tickets' list."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def ok_ticket(t: TicketText) -> bool:
    words = len(t.body.split())
    banned = ("known issue", "KI-0")
    return bool(t.subject.strip()) and 4 <= words <= 300 and not any(b in t.body for b in banned)


def generate_for_scenario(llm: OllamaLLM, s: dict, seed: int) -> list[dict]:
    """Generate s['n'] tickets in batches of <=5, retrying for rejected/missing ones."""
    rng = random.Random(f"{seed}-{s['id']}")
    tickets: list[dict] = []
    attempt = 0
    while len(tickets) < s["n"] and attempt < 6:
        k = min(5, s["n"] - len(tickets))
        styles = rng.sample(STYLES, k)
        region = rng.choice(s["region"]) if s["region"] else rng.choice(REGIONS + ["any"])
        batch, _ = llm.structured(generation_prompt(s, styles, region), TicketBatch,
                                  temperature=0.9, seed=seed + attempt, think=False)
        seen = {t["body"] for t in tickets}
        for t, style in zip(batch.tickets, styles):
            if ok_ticket(t) and t.body not in seen:
                tickets.append({"subject": t.subject.strip(), "body": t.body.strip(),
                                "style": style, "region": region})
                seen.add(t.body)
        attempt += 1
    if len(tickets) < s["n"]:
        print(f"  WARNING {s['id']}: only {len(tickets)}/{s['n']} tickets after {attempt} attempts")
    return tickets[: s["n"]]


# ---------- outputs ----------

def to_rows(scenarios: list[dict], generated: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for s in sorted(scenarios, key=lambda x: x["id"]):
        for i, t in enumerate(generated[s["id"]]):
            rows.append({
                "ticket_id": "",
                "scenario_id": s["id"], "variant": i, "split": s["split"], "type": s["type"],
                "subject": t["subject"], "body": t["body"], "text": f"{t['subject']}\n\n{t['body']}",
                "category": s["category"], "known_or_new": s["known_or_new"],
                "matched_issue_id": s["matched_issue_id"], "relevant_chunk_ids": s["relevant_chunk_ids"],
                "reference_answer": s["reference_answer"], "should_escalate": s["should_escalate"],
                "region": t["region"], "style": t["style"],
            })
    for i, r in enumerate(rows):
        r["ticket_id"] = f"T-{i + 1:04d}"
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)  # bytes, so Windows does not turn \n into \r\n and break the hash
    return hashlib.sha256(data).hexdigest()


def write_review_csv(rows: list[dict], scenarios: list[dict], n: int, seed: int) -> None:
    """Stratified sample (by scenario type) for the team to check labels."""
    df = pd.DataFrame(rows)
    briefs = {s["id"]: s["brief"] for s in scenarios}
    per_type = (df["type"].value_counts(normalize=True) * n).round().clip(lower=2).astype(int)
    parts = [g.sample(min(len(g), per_type[t]), random_state=seed) for t, g in df.groupby("type")]
    sample = pd.concat(parts).head(n).sort_values("ticket_id")
    sample["scenario_brief"] = sample["scenario_id"].map(briefs)
    cols = ["ticket_id", "split", "type", "subject", "body", "scenario_brief", "category", "known_or_new",
            "matched_issue_id", "should_escalate", "reference_answer"]
    out = sample[cols].copy()
    out["text_fits_scenario (y/n)"] = ""
    out["labels_ok (y/n)"] = ""
    out["notes"] = ""
    REVIEW.mkdir(parents=True, exist_ok=True)
    out.to_csv(REVIEW / "synthetic_tickets_review.csv", index=False, encoding="utf-8-sig")
    print(f"review CSV: {len(out)} tickets -> data/review/synthetic_tickets_review.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="validate and split only; no LLM calls")
    parser.add_argument("--refreeze", action="store_true", help="allow the frozen test split to change")
    args = parser.parse_args()
    cfg = load_config()
    ds = cfg.datasets
    seed = ds["seed"]
    spec = yaml.safe_load((SYN / "scenarios.yaml").read_text(encoding="utf-8"))

    known = build_known_issues(spec)
    scenarios = build_scenarios(spec, known)
    validate(scenarios, known, ds["labels"])
    split_scenarios(scenarios, seed)
    with open(SYN / "known_issues.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(k, ensure_ascii=False) + "\n" for k in known)
    n_total = sum(s["n"] for s in scenarios)
    n_known = sum(s["n"] for s in scenarios if s["known_or_new"] == "known")
    print(f"{len(known)} known issues, {len(scenarios)} scenarios, {n_total} tickets planned "
          f"({100 * n_known / n_total:.0f}% known); KB chunks: {len(chunk_kb_dir(KB))}")
    if args.dry_run:
        return

    llm = OllamaLLM(cfg, "fast")
    generated = {}
    for i, s in enumerate(scenarios):
        generated[s["id"]] = generate_for_scenario(llm, s, seed)
        print(f"  [{i + 1}/{len(scenarios)}] {s['id']}: {len(generated[s['id']])} tickets", flush=True)
    rows = to_rows(scenarios, generated)

    TICKETS.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in ("train", "dev", "test"):
        part = [r for r in rows if r["split"] == split]
        if split == "test":
            frozen = TICKETS / "test.sha256"
            new_hash = hashlib.sha256("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                              for r in part).encode("utf-8")).hexdigest()
            if frozen.exists() and frozen.read_text().strip() != new_hash and not args.refreeze:
                raise SystemExit("test split would change but is frozen; rerun with --refreeze if intended")
        hashes[split] = write_jsonl(TICKETS / f"{split}.jsonl", part)
    (TICKETS / "test.sha256").write_text(hashes["test"] + "\n")
    write_review_csv(rows, scenarios, ds["synthetic"]["review_rows"], seed)

    write_summary(pd.DataFrame(rows))


def write_summary(df: pd.DataFrame) -> None:
    """Counts per split x type / label / known-new -> results/tables/synthetic_summary.md."""
    order = ["train", "dev", "test"]

    def table(col: str) -> str:
        t = pd.crosstab(df[col], df["split"])[order]
        t["total"] = t.sum(axis=1)
        return t.reset_index().to_markdown(index=False)

    lines = ["# Track A synthetic tickets", "", "Generated by `scripts/generate_synthetic.py`.", "",
             f"Total: {len(df)} tickets from {df['scenario_id'].nunique()} scenarios; "
             f"{100 * (df['known_or_new'] == 'known').mean():.1f}% known.", ""]
    for col in ("type", "category", "known_or_new", "should_escalate"):
        lines += [f"## By {col}", "", table(col), ""]
    path = ROOT / "results" / "tables" / "synthetic_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

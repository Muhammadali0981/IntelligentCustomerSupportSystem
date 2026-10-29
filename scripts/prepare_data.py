"""Map labels, deduplicate, subsample and split the public datasets.

Run:  py -3.11 scripts/prepare_data.py
Reads data/raw/, writes data/processed/ and data/review/. Fixed seed from
configs/datasets.yaml. Deduplication (cosine > threshold) runs on the GPU with
no LLM loaded, before any split, so paraphrases cannot leak across splits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, Config, load_config  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
REVIEW = ROOT / "data" / "review"


# ---------- classification: Bitext + Tobi-Bueck ----------

def load_bitext(ds: dict) -> pd.DataFrame:
    """Bitext rows with our label. Intent map overrides category map."""
    df = pd.read_parquet(RAW / "bitext" / "default__train.parquet")
    b = ds["bitext"]
    df["label"] = df["intent"].map(b["map_intents"]).fillna(df["category"].map(b["map_categories"]))
    unmapped = df[df["label"].isna()][["category", "intent"]].drop_duplicates()
    if len(unmapped):
        raise ValueError(f"unmapped Bitext categories/intents: {unmapped.values.tolist()}")
    return pd.DataFrame({
        "source": "bitext", "text": df["instruction"].str.strip(), "label": df["label"],
        "orig_category": df["category"], "orig_intent": df["intent"],
    })


def load_tobi(ds: dict) -> pd.DataFrame:
    """English Tobi-Bueck rows from one file; text = subject + body only."""
    t = ds["tobi_bueck"]
    df = pd.read_parquet(RAW / "tobi_bueck" / "default__train.parquet")
    df = df[df["language"] == t["language"]].copy()
    df["label"] = df["queue"].map(t["map_queues"])
    df = df[df["label"].notna()]
    text = (df["subject"].fillna("").str.strip() + "\n\n" + df["body"].fillna("").str.strip()).str.strip()
    return pd.DataFrame({
        "source": "tobi_bueck", "text": text, "label": df["label"],
        "orig_category": df["queue"], "orig_intent": df["type"],
        "subject": df["subject"], "body": df["body"],
    })


def dedup(df: pd.DataFrame, cfg: Config, threshold: float, seed: int) -> pd.DataFrame:
    """Greedy near-duplicate removal: keep a row unless an already-kept row has cosine > threshold.

    Rows are visited in a seeded random order. Runs on GPU (no LLM loaded) in blocks.
    """
    import torch

    from src.llm.ollama_client import unload_all
    from src.retrieval.embedding import Embedder

    unload_all(cfg)
    device = cfg.embedder()["build_device"] if torch.cuda.is_available() else "cpu"
    emb = Embedder(cfg, device=device)
    df = df.drop_duplicates("text").sample(frac=1.0, random_state=seed).reset_index(drop=True)
    vecs = torch.from_numpy(emb.encode_docs(df["text"].tolist(), show_progress=True)).to(device)
    emb.free()

    kept: list[int] = []
    block = 1024
    for start in range(0, len(df), block):
        idx = list(range(start, min(start + block, len(df))))
        blk = vecs[idx]
        if kept:  # vs. everything kept so far
            ok = (blk @ vecs[kept].T).max(dim=1).values <= threshold
        else:
            ok = torch.ones(len(idx), dtype=torch.bool, device=device)
        sims = blk @ blk.T  # within the block, sequentially
        local_kept: list[int] = []
        for i in range(len(idx)):
            if ok[i] and (not local_kept or sims[i, local_kept].max() <= threshold):
                local_kept.append(i)
        kept += [idx[i] for i in local_kept]
    del vecs
    torch.cuda.empty_cache()
    return df.iloc[sorted(kept)].reset_index(drop=True)


def sample_groups(df: pd.DataFrame, by: str, n: dict[str, int] | int, seed: int) -> pd.DataFrame:
    """Sample up to n rows per group (n can differ per group)."""
    parts = [g.sample(min(len(g), n if isinstance(n, int) else n[key]), random_state=seed)
             for key, g in df.groupby(by)]
    return pd.concat(parts)


def subsample(df: pd.DataFrame, ds: dict, seed: int) -> pd.DataFrame:
    """~N per Bitext intent; fixed counts per label for Tobi-Bueck."""
    bit = sample_groups(df[df.source == "bitext"], "orig_intent", ds["bitext"]["per_intent"], seed)
    tobi = sample_groups(df[df.source == "tobi_bueck"], "label", ds["tobi_bueck"]["sample_per_label"], seed)
    return pd.concat([bit, tobi], ignore_index=True)


def stratified_split(df: pd.DataFrame, fracs: dict[str, float], seed: int, by: list[str]) -> pd.DataFrame:
    """Add a 'split' column, stratified on the given columns."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = ""
    for _, g in df.groupby(by):
        order = rng.permutation(g.index.to_numpy())
        n_train = round(len(order) * fracs["train"])
        n_dev = round(len(order) * fracs["dev"])
        df.loc[order[:n_train], "split"] = "train"
        df.loc[order[n_train:n_train + n_dev], "split"] = "dev"
        df.loc[order[n_train + n_dev:], "split"] = "test"
    return df


def prepare_classification(cfg: Config) -> pd.DataFrame:
    ds, seed = cfg.datasets, cfg.datasets["seed"]
    raw = pd.concat([load_bitext(ds), load_tobi(ds)], ignore_index=True)
    print(f"classification: {len(raw)} mapped rows before dedup "
          f"({raw.source.value_counts().to_dict()})")
    deduped = dedup(raw, cfg, ds["bitext"]["dedup_cosine"], seed)
    print(f"  after dedup (cosine > {ds['bitext']['dedup_cosine']}): {len(deduped)} "
          f"({deduped.source.value_counts().to_dict()})")
    sub = subsample(deduped, ds, seed)
    sub = stratified_split(sub, ds["classification_split"], seed, by=["source", "label"])
    sub.insert(0, "id", [f"{s[:3]}-{i:05d}" for i, s in enumerate(sub["source"])])
    out = OUT / "classification"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["id", "source", "text", "label", "orig_category", "orig_intent", "split"]
    for split in ("train", "dev", "test"):
        sub[sub.split == split][cols].to_parquet(out / f"{split}.parquet", index=False)
    stats = {"raw": int(len(raw)), "after_dedup": int(len(deduped)),
             "removed_by_dedup": raw.source.value_counts().sub(deduped.source.value_counts()).to_dict()}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    # Human spot-check of noisy Tobi-Bueck queue labels
    tobi = sub[sub.source == "tobi_bueck"]
    n = ds["tobi_bueck"]["review_rows"]
    per_label = {k: round(n * v / len(tobi)) for k, v in tobi["label"].value_counts().items()}
    review = sample_groups(tobi, "label", per_label, seed)
    review = review[["id", "subject", "body", "orig_category", "label"]].rename(
        columns={"orig_category": "queue", "label": "mapped_label"})
    review["label_ok (y/n)"] = ""
    review["correct_label_if_n"] = ""
    review["notes"] = ""
    REVIEW.mkdir(parents=True, exist_ok=True)
    review.to_csv(REVIEW / "tobi_bueck_label_spotcheck.csv", index=False, encoding="utf-8-sig")
    print(f"  review CSV: {len(review)} Tobi-Bueck rows -> data/review/tobi_bueck_label_spotcheck.csv")
    return sub


# ---------- WixQA, CLINC150, RAGBench ----------

def prepare_wixqa(cfg: Config) -> dict:
    out = OUT / "wixqa"
    out.mkdir(parents=True, exist_ok=True)
    kb = pd.read_parquet(RAW / "wixqa" / "wix_kb_corpus__train.parquet")
    kb[["id", "url", "title", "contents", "article_type"]].to_parquet(out / "kb_corpus.parquet", index=False)

    tests = []
    for c in cfg.datasets["wixqa"]["test_configs"]:
        q = pd.read_parquet(RAW / "wixqa" / f"{c}__train.parquet")
        q.insert(0, "qid", [f"{c.split('_')[1][:3]}-{i:03d}" for i in range(len(q))])
        q.insert(1, "subset", c)
        tests.append(q)
    test = pd.concat(tests, ignore_index=True)
    test["article_ids"] = test["article_ids"].apply(list)
    test.to_parquet(out / "test_real.parquet", index=False)

    dev = pd.read_parquet(RAW / "wixqa" / f"{cfg.datasets['wixqa']['dev_config']}__train.parquet")
    dev.insert(0, "qid", [f"syn-{i:04d}" for i in range(len(dev))])
    dev["article_ids"] = dev["article_ids"].apply(list)
    dev.to_parquet(out / "dev_synthetic.parquet", index=False)

    types = kb.article_type.value_counts().to_dict()
    kb_types = kb.set_index("id")["article_type"]
    gold_types = pd.Series([kb_types[a] for ids in test.article_ids for a in ids]).value_counts().to_dict()
    stats = {"kb_articles": len(kb), "article_types": types, "known_issue_count": types.get("known_issue", 0),
             "test_questions": len(test), "dev_questions": len(dev), "test_gold_article_types": gold_types}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"wixqa: {stats}")
    return stats


def prepare_clinc(cfg: Config) -> None:
    names = json.loads((RAW / "clinc" / "plus__labels.json").read_text())["intent"]
    oos_id = names.index(cfg.datasets["clinc"]["oos_label"])
    out = OUT / "clinc"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):  # validation tunes thresholds, test evaluates
        df = pd.read_parquet(RAW / "clinc" / f"plus__{split}.parquet")
        df["intent_name"] = df["intent"].map(lambda i: names[i])
        df["is_oos"] = df["intent"] == oos_id
        df.to_parquet(out / f"{split}.parquet", index=False)
        print(f"clinc/{split}: {len(df)} rows, oos={int(df.is_oos.sum())}")


def summary_table(clf: pd.DataFrame) -> None:
    """Row counts per source / split / label -> results/tables/data_summary.md."""
    pivot = clf.pivot_table(index=["source", "label"], columns="split", values="id", aggfunc="count",
                            fill_value=0)[["train", "dev", "test"]]
    pivot["total"] = pivot.sum(axis=1)
    wix = json.loads((OUT / "wixqa" / "stats.json").read_text())
    lines = ["# Data summary (public datasets)", "", "Generated by `scripts/prepare_data.py`.", "",
             "## Classification (Bitext + Tobi-Bueck, after dedup + subsample)", "",
             pivot.reset_index().to_markdown(index=False), "",
             f"Totals: train {int(pivot['train'].sum())}, dev {int(pivot['dev'].sum())}, "
             f"test {int(pivot['test'].sum())}.", "", "## WixQA", "",
             f"- KB articles: {wix['kb_articles']} ({wix['article_types']})",
             f"- known_issue articles: **{wix['known_issue_count']}**",
             f"- Real test questions (expertwritten + simulated): {wix['test_questions']}; "
             f"gold article types: {wix['test_gold_article_types']}",
             f"- Synthetic dev questions: {wix['dev_questions']}", ""]
    for split in ("validation", "test"):
        c = pd.read_parquet(OUT / "clinc" / f"{split}.parquet")
        lines.append(f"- CLINC150 plus {split}: {len(c)} rows ({int(c.is_oos.sum())} oos)")
    for cfg_name in ("techqa", "emanual"):
        counts = {s: len(pd.read_parquet(RAW / "ragbench" / f"{cfg_name}__{s}.parquet"))
                  for s in ("train", "validation", "test")}
        lines.append(f"- RAGBench {cfg_name}: {counts}")
    path = ROOT / "results" / "tables" / "data_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    cfg = load_config()
    clf = prepare_classification(cfg)
    prepare_wixqa(cfg)
    prepare_clinc(cfg)
    summary_table(clf)


if __name__ == "__main__":
    main()

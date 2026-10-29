"""Download every public dataset into data/raw/ as parquet. Safe to re-run:
files that already exist are skipped (use --force to re-download).

Run:  py -3.11 scripts/download_data.py [--optional] [--force]
Writes data/raw/<name>/<config>__<split>.parquet and data/raw/MANIFEST.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_config  # noqa: E402

RAW = ROOT / "data" / "raw"
HF_CACHE = RAW / ".hf_cache"


def save_dataset(name: str, ds_id: str, config: str | None = None, force: bool = False, **kwargs) -> list[dict]:
    """Load one HF dataset (all splits) and write each split to parquet."""
    from datasets import load_dataset

    out_dir = RAW / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = config or "default"
    existing = sorted(out_dir.glob(f"{tag}__*.parquet"))
    if existing and not force:
        print(f"  skip {name}/{tag}: {len(existing)} file(s) present")
        return [_describe(p) for p in existing]

    from datasets import ClassLabel

    dsd = load_dataset(ds_id, config, cache_dir=str(HF_CACHE), **kwargs)
    entries = []
    # Parquet drops ClassLabel names (e.g. CLINC intents become ints); keep them alongside.
    first = next(iter(dsd.values()))
    names = {col: feat.names for col, feat in first.features.items() if isinstance(feat, ClassLabel)}
    if names:
        (out_dir / f"{tag}__labels.json").write_text(json.dumps(names, indent=1))
    for split, ds in dsd.items():
        path = out_dir / f"{tag}__{split}.parquet"
        ds.to_parquet(str(path))
        entries.append(_describe(path))
        print(f"  {name}/{tag}/{split}: {ds.num_rows} rows, columns={ds.column_names}")
    return entries


def save_file(name: str, ds_id: str, filename: str, force: bool = False) -> list[dict]:
    """Download a single raw file from a dataset repo (for repos without a loader)."""
    from huggingface_hub import hf_hub_download

    out_dir = RAW / name
    if (out_dir / filename).exists() and not force:
        print(f"  skip {name}/{filename}: present")
    else:
        hf_hub_download(ds_id, filename, repo_type="dataset", local_dir=str(out_dir))
        print(f"  {name}/{filename}: downloaded")
    return [{"path": str((out_dir / filename).relative_to(ROOT)), "rows": None, "columns": None}]


def _describe(path: Path) -> dict:
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(path).metadata
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "rows": meta.num_rows,
            "columns": meta.schema.to_arrow_schema().names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optional", action="store_true", help="also fetch optional datasets")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    ds = load_config().datasets
    f = args.force
    manifest: dict[str, list[dict]] = {}

    print("Bitext")
    manifest["bitext"] = save_dataset("bitext", ds["bitext"]["id"], force=f)
    print("Tobi-Bueck (one file only)")
    manifest["tobi_bueck"] = save_dataset("tobi_bueck", ds["tobi_bueck"]["id"], force=f,
                                          data_files=ds["tobi_bueck"]["data_file"])
    print("WixQA")
    manifest["wixqa"] = []
    for cfg in [ds["wixqa"]["corpus_config"], *ds["wixqa"]["test_configs"], ds["wixqa"]["dev_config"]]:
        manifest["wixqa"] += save_dataset("wixqa", ds["wixqa"]["id"], cfg, force=f)
    print("CLINC150")
    manifest["clinc"] = save_dataset("clinc", ds["clinc"]["id"], ds["clinc"]["config"], force=f)
    print("RAGBench")
    manifest["ragbench"] = []
    for cfg in ds["ragbench"]["configs"]:
        manifest["ragbench"] += save_dataset("ragbench", ds["ragbench"]["id"], cfg, force=f)

    if args.optional:
        opt = ds["optional"]
        print("MakTek FAQs")
        manifest["maktek_faqs"] = save_dataset("maktek_faqs", opt["maktek_faqs"]["id"], force=f)
        print("TechQA-RAG-Eval (train.json only; contexts are inline)")
        manifest["techqa_rag_eval"] = save_file("techqa_rag_eval", opt["techqa_rag_eval"]["id"], "train.json", f)
        print("banking77 (parquet branch)")
        try:
            manifest["banking77"] = save_dataset("banking77", opt["banking77"]["id"], force=f,
                                                 revision=opt["banking77"]["revision"])
        except Exception as e:  # optional: report and skip, never downgrade the env
            print(f"  banking77 FAILED, skipped: {type(e).__name__}: {e}")

    old = json.loads((RAW / "MANIFEST.json").read_text()) if (RAW / "MANIFEST.json").exists() else {}
    (RAW / "MANIFEST.json").write_text(json.dumps({**old, **manifest}, indent=2))
    print(f"\nWrote {RAW / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()

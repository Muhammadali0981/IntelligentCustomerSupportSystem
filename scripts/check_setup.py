"""Verify the environment: Ollama + models, structured output, thinking toggle,
embedder, reranker, Chroma persistence and dataset IDs.

Run:  py -3.11 scripts/check_setup.py [--profile home]
Exit code is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel  # noqa: E402

from src.config import MODELS_CACHE, ROOT, Config, load_config  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str) -> Callable:
    """Decorator: run a check, record pass/fail and a detail string."""
    def wrap(fn: Callable[..., str]) -> Callable[..., None]:
        def run(*args, **kwargs) -> None:
            t0 = time.perf_counter()
            try:
                detail = fn(*args, **kwargs)
                RESULTS.append((name, True, f"{detail} ({time.perf_counter() - t0:.1f}s)"))
            except Exception as e:  # report every failure, keep going
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            ok, msg = RESULTS[-1][1], RESULTS[-1][2]
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
        return run
    return wrap


def gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


@check("system")
def check_system() -> str:
    import torch
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else "none"
    free_gb = shutil.disk_usage(Path.cwd()).free / 1e9
    return (f"{platform.system()} {platform.release()}, Python {platform.python_version()}, "
            f"torch {torch.__version__} CUDA={cuda} ({gpu}), disk free {free_gb:.0f} GB, "
            f"GPU mem used now {gpu_mem_used_mib()} MiB")


@check("ollama reachable + models present")
def check_ollama_models(cfg: Config) -> str:
    import ollama
    client = ollama.Client(host=cfg.profile["ollama"]["host"])
    present = {m.model for m in client.list().models}
    needed = {cfg.route(r)["tag"] for r in cfg.profile["routes"]}
    missing = needed - present
    if missing:
        raise RuntimeError(f"missing models: {sorted(missing)}; present: {sorted(present)}")
    return f"Ollama {client._client.get('/api/version').json()['version']}, models: {sorted(needed)}"


class Triage(BaseModel):
    category: str
    confidence: float


@check("JSON-structured call (fast route)")
def check_structured(cfg: Config) -> str:
    import ollama
    r = cfg.route("fast")
    client = ollama.Client(host=cfg.profile["ollama"]["host"])
    resp = client.chat(
        model=r["tag"], think=False,
        messages=[{"role": "user", "content":
                   "Classify this ticket as billing, account_access, technical, shipping or other: "
                   "'I was charged twice for my order.' Reply as JSON."}],
        format=Triage.model_json_schema(),
        options={"temperature": 0, "num_ctx": r["num_ctx"]},
    )
    parsed = Triage.model_validate_json(resp.message.content)
    return f"{parsed.model_dump()}"


@check("thinking off vs on")
def check_thinking(cfg: Config) -> str:
    import ollama
    client = ollama.Client(host=cfg.profile["ollama"]["host"])
    msgs = [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}]
    details = []
    for route in ("fast", "reasoning"):
        r = cfg.route(route)
        resp = client.chat(model=r["tag"], messages=msgs, think=r["think"],
                           options={"temperature": 0, "num_ctx": r["num_ctx"]})
        thinking = resp.message.thinking or ""
        if r["think"] and not thinking:
            raise RuntimeError(f"{route}: think=True returned no thinking text")
        if not r["think"] and thinking:
            raise RuntimeError(f"{route}: think=False still returned thinking text")
        details.append(f"{route}(think={r['think']}): answer={resp.message.content.strip()!r}, "
                       f"thinking_chars={len(thinking)}, eval_tokens={resp.eval_count}")
    ps = client.ps().models
    vram = ", ".join(f"{m.model} size_vram={m.size_vram / 2**30:.2f} GiB ctx={m.context_length}" for m in ps)
    details.append(f"loaded: {vram}; nvidia-smi used={gpu_mem_used_mib()} MiB")
    return " | ".join(details)


@check("embedder loads (CPU)")
def check_embedder(cfg: Config) -> str:
    from sentence_transformers import SentenceTransformer
    e = cfg.embedder()
    model = SentenceTransformer(e["hf_id"], device="cpu", cache_folder=str(MODELS_CACHE))
    kw = {"prompt_name": e["query_prompt_name"]} if e.get("query_prompt_name") else {}
    q = model.encode(["How do I reset my password?"], normalize_embeddings=True, **kw)
    d = model.encode(["To reset your password, click 'Forgot password' on the login page.",
                      "Shipping takes 3-5 business days."], normalize_embeddings=True)
    sims = (q @ d.T)[0]
    if not sims[0] > sims[1]:
        raise RuntimeError(f"relevant doc not ranked first: {sims}")
    return f"{e['hf_id']} dim={model.get_embedding_dimension()} sims={[round(float(s), 3) for s in sims]}"


@check("reranker loads (CPU)")
def check_reranker(cfg: Config) -> str:
    from sentence_transformers import CrossEncoder
    out = []
    for key in cfg.models["rerankers"]:
        rr = cfg.reranker(key)
        model = CrossEncoder(rr["hf_id"], device="cpu", cache_folder=str(MODELS_CACHE))
        scores = model.predict([("reset password", "Click 'Forgot password' to reset it."),
                                ("reset password", "Shipping takes 3-5 business days.")])
        if not scores[0] > scores[1]:
            raise RuntimeError(f"{key}: relevant pair not scored higher: {scores}")
        out.append(f"{key} scores={[round(float(s), 3) for s in scores]}")
    return "; ".join(out)


@check("Chroma persists to disk")
def check_chroma() -> str:
    import chromadb
    # A fixed folder rather than a temp dir: on Windows, Chroma keeps its index
    # files open, so deleting a temp dir in the same process fails.
    path = ROOT / "data" / "index" / "_setup_check"
    stamp = str(time.time())
    c1 = chromadb.PersistentClient(path=str(path))
    c1.get_or_create_collection("check").upsert(ids=["a"], documents=[stamp], embeddings=[[0.1, 0.2, 0.3]])
    del c1
    from chromadb.api.shared_system_client import SharedSystemClient
    SharedSystemClient.clear_system_cache()  # force a real reopen from disk
    got = chromadb.PersistentClient(path=str(path)).get_collection("check").get(ids=["a"])
    if got["documents"] != [stamp]:
        raise RuntimeError(f"unexpected: {got}")
    return f"chromadb {chromadb.__version__}: wrote, reopened from {path.relative_to(ROOT)}, read back"


@check("dataset IDs resolve on Hugging Face")
def check_datasets(cfg: Config) -> str:
    from huggingface_hub import HfApi
    ds = cfg.datasets
    ids = [ds[k]["id"] for k in ("bitext", "tobi_bueck", "wixqa", "clinc", "ragbench")]
    ids += [v["id"] for v in ds["optional"].values()]
    api = HfApi()
    for i in ids:
        api.dataset_info(i)
    return f"{len(ids)} OK: {', '.join(ids)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()
    cfg = load_config(args.profile)
    print(f"Profile: {cfg.profile_name}\n")

    check_system()
    check_ollama_models(cfg)
    check_structured(cfg)
    check_thinking(cfg)
    check_embedder(cfg)
    check_reranker(cfg)
    check_chroma()
    check_datasets(cfg)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

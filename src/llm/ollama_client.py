"""Thin wrapper around the Ollama chat API with caching, timing and
Pydantic-validated structured output.

    from src.llm.ollama_client import OllamaLLM
    llm = OllamaLLM(cfg, "fast")
    res = llm.chat([{"role": "user", "content": "hi"}])
    obj, res = llm.structured(messages, MySchema)   # validated, one retry
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from src.config import Config
from src.llm import cache

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResult:
    """One LLM call: output text, optional thinking, token counts and latency."""

    content: str
    thinking: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str
    cached: bool = False


class OllamaLLM:
    """A route (fast / reasoning / judge ...) bound to its model and settings."""

    def __init__(self, cfg: Config, route: str, keep_alive: str | int | None = None):
        self.route_name = route
        self.route = cfg.route(route)
        self.tag: str = self.route["tag"]
        self.keep_alive = keep_alive if keep_alive is not None else cfg.profile["ollama"]["keep_alive"]
        self.client = ollama.Client(host=cfg.profile["ollama"]["host"])

    def chat(self, messages: list[dict[str, str]], *, schema: dict | None = None, think: bool | str | None = None,
             temperature: float = 0.0, num_predict: int | None = None, seed: int | None = None,
             use_cache: bool = True) -> LLMResult:
        """Send one chat request. `think=None` uses the route's default."""
        think = self.route["think"] if think is None else think
        options: dict[str, Any] = {"temperature": temperature, "num_ctx": self.route["num_ctx"]}
        if num_predict is not None:
            options["num_predict"] = num_predict
        if seed is not None:
            options["seed"] = seed
        key = cache.make_key({"model": self.tag, "messages": messages, "format": schema,
                              "think": think, "options": options})
        if use_cache and (hit := cache.get(key)):
            return LLMResult(**{**hit, "cached": True})

        t0 = time.perf_counter()
        resp = self.client.chat(model=self.tag, messages=messages, format=schema, think=think,
                                options=options, keep_alive=self.keep_alive)
        result = LLMResult(
            content=resp.message.content or "",
            thinking=resp.message.thinking or "",
            prompt_tokens=resp.prompt_eval_count or 0,
            completion_tokens=resp.eval_count or 0,
            latency_s=time.perf_counter() - t0,
            model=self.tag,
        )
        if use_cache:
            cache.put(key, asdict(result))
        return result

    def structured(self, messages: list[dict[str, str]], model_cls: type[T], **kwargs) -> tuple[T, LLMResult]:
        """Chat with a JSON schema; validate with Pydantic and retry once on failure."""
        schema = model_cls.model_json_schema()
        res = self.chat(messages, schema=schema, **kwargs)
        try:
            return model_cls.model_validate_json(res.content), res
        except ValidationError as err:
            retry = messages + [
                {"role": "assistant", "content": res.content},
                {"role": "user", "content": f"That output was invalid ({err.error_count()} errors). "
                                            "Reply again with JSON that matches the schema exactly."},
            ]
            res = self.chat(retry, schema=schema, **kwargs)
            return model_cls.model_validate_json(res.content), res

    def unload(self) -> None:
        """Free VRAM (e.g. before bulk GPU embedding)."""
        self.client.generate(model=self.tag, prompt="", keep_alive=0)


def unload_all(cfg: Config) -> None:
    """Unload every model Ollama currently holds in memory."""
    client = ollama.Client(host=cfg.profile["ollama"]["host"])
    for m in client.ps().models:
        client.generate(model=m.model, prompt="", keep_alive=0)

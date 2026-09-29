"""Routing from a model name to the endpoint that serves it.

Every model speaks the OpenAI chat-completions API — open weights through a
local vLLM server, closed weights through OpenRouter — so there is one code
path and `models.yaml` decides where each call goes.

Two things follow from routing being config rather than code. Model names in
logs are the config's aliases, which stay stable when an upstream id is
renamed; and an unknown name is an error rather than a guess, because guessing
wrong means either a confusing connection failure or an unintended bill.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import yaml
from openai import OpenAI

# Resolved against this file, not the working directory: everything else here
# is opened relative to the vendor directory, but a run that cannot find its
# routing table has no sensible fallback, so this one is made unmissable.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "models.yaml"


@dataclass(frozen=True)
class ModelRoute:
    alias: str      # what the caller asked for, and what goes in the logs
    backend: str    # "vllm" | "openrouter" | any other key in models.yaml
    model_id: str   # the id actually sent on the wire

    def __str__(self) -> str:
        return f"{self.backend}:{self.model_id}"


class ModelRegistry:
    def __init__(self, config_path: Path | str | None = None):
        self.config_path = Path(
            config_path
            or os.environ.get("LAYERED_VQA_MODELS_CONFIG", DEFAULT_CONFIG_PATH)
        )
        self._lock = threading.Lock()
        self._clients: dict[str, OpenAI] = {}
        self._announced: set[str] = set()
        self._config: dict | None = None

    @property
    def config(self) -> dict:
        if self._config is None:
            if not self.config_path.exists():
                raise FileNotFoundError(
                    f"Model routing config not found at {self.config_path}. "
                    "Set LAYERED_VQA_MODELS_CONFIG or restore models.yaml."
                )
            self._config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        return self._config

    @property
    def backends(self) -> dict:
        return self.config.get("backends") or {}

    @property
    def models(self) -> dict:
        return self.config.get("models") or {}

    def resolve(self, name: str) -> ModelRoute:
        """Map a model name onto the backend that serves it.

        Accepts either an alias from `models.yaml` or an inline `backend:id`
        for a model that has not been added to the config yet.
        """
        if ":" in name:
            backend, _, model_id = name.partition(":")
            if backend not in self.backends:
                raise ValueError(
                    f"Unknown backend {backend!r} in model {name!r}. "
                    f"Known backends: {', '.join(sorted(self.backends))}."
                )
            return self._route(name, backend, model_id)

        entry = self.models.get(name)
        if entry is None:
            raise ValueError(
                f"Unknown model {name!r}. Add it to {self.config_path.name}, or "
                f"pass it inline as backend:id (e.g. openrouter:openai/gpt-4o-mini). "
                f"Configured models: {', '.join(sorted(self.models)) or '(none)'}."
            )
        backend = entry.get("backend")
        if backend not in self.backends:
            raise ValueError(
                f"Model {name!r} names backend {backend!r}, which {self.config_path.name} "
                f"does not define. Known backends: {', '.join(sorted(self.backends))}."
            )
        return self._route(name, backend, entry.get("model", name))

    def _route(self, alias: str, backend: str, model_id: str) -> ModelRoute:
        route = ModelRoute(alias, backend, model_id)
        self._announce(route)
        return route

    def _announce(self, route: ModelRoute) -> None:
        """Say once, per model, where its calls are going.

        Silent routing is the failure mode worth spending a print on: the same
        command that used to hit a free local server can now bill an account.
        """
        with self._lock:
            if route.alias in self._announced:
                return
            self._announced.add(route.alias)
        print(f"[models] {route.alias} -> {route}")

    def client_for(self, backend: str) -> OpenAI:
        with self._lock:
            if backend not in self._clients:
                self._clients[backend] = self._build_client(backend)
            return self._clients[backend]

    def _build_client(self, backend: str) -> OpenAI:
        spec = self.backends[backend]

        base_url = os.environ.get(spec.get("base_url_env", ""), "") or spec.get("base_url")
        if not base_url:
            raise ValueError(
                f"Backend {backend!r} has no base_url: set {spec.get('base_url_env')} "
                f"or give it a `base_url` in {self.config_path.name}."
            )

        api_key_env = spec.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "") or spec.get("api_key")
        if not api_key:
            raise ValueError(
                f"Backend {backend!r} needs an API key: set {api_key_env}."
            )

        return OpenAI(base_url=base_url, api_key=api_key)

    def extra_headers(self, backend: str) -> dict:
        return dict(self.backends[backend].get("extra_headers") or {})


registry = ModelRegistry()

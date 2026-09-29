"""Generation settings for the three agents, read from `agents.yaml`.

Every model call in the simulator is made by one of three agents: the
*assistant* (the model under test), the *user* (the simulated user revealing
shards), and the *judge* (which grades the finished conversation). What each of
them sends — sampling settings, token budget, timeout, retries — used to be literals
spread across four simulators, the user agent and the judge, which is why the
same knob had three different values depending on which entry point you
launched. `agents.yaml` is now the single place that says it.

Two things follow from these being config rather than code. An unknown key is
an error, never an ignored line, because a misspelled `max_tokens` that "worked"
would quietly run a whole experiment on the provider's default budget. And the
settings actually in force are announced once per agent, for the same reason
`model_registry` announces routing: temperature decides what the numbers in a
results table mean, and it should never be invisible.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import yaml

# Resolved against this file, not the working directory, exactly like
# `models.yaml`: everything else here is opened relative to the vendor
# directory, but the routing and generation configs have no sensible fallback
# location to search.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "agents.yaml"

AGENTS = ("assistant", "user", "judge")
SETTINGS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "reasoning_max_tokens",
    "timeout",
    "max_retries",
)

# Shipped fallback settings, kept in sync with `agents.yaml`. Each agent falls
# back to these values setting by setting when a line is omitted. They also
# stand in for the whole file when it is missing from its default location, so
# a partial checkout still runs; an explicitly configured path that does not
# exist is an error instead.
BUILTIN = {
    "agents": {
        "assistant": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "max_tokens": 1024,
            "reasoning_max_tokens": 16384,
            "timeout": 600,
            "max_retries": 3,
        },
        "user": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "max_tokens": 1024,
            "reasoning_max_tokens": 16384,
            "timeout": 300,
            "max_retries": 3,
        },
        "judge": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "max_tokens": 1024,
            "reasoning_max_tokens": 16384,
            "timeout": 600,
            "max_retries": 3,
        },
    },
    "reasoning_models": ["o1", "o3", "deepseek-r1", "qwen3", "Qwen3", "minimax"],
}


@dataclass(frozen=True)
class AgentParams:
    """One agent's settings, ready to be splatted into `generate`."""

    agent: str
    temperature: float = 1.0
    # Both nucleus knobs are optional: None means the parameter is left off the
    # request entirely, so the provider's own default applies. There is no
    # portable "disabled" value for top_k — vLLM spells it -1, OpenRouter 0 —
    # so it is omitted rather than guessed at.
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    reasoning_max_tokens: int | None = None
    timeout: int = 120
    max_retries: int = 3
    reasoning_models: tuple[str, ...] = ()

    def is_reasoning_model(self, model: str | None) -> bool:
        return bool(model) and any(pattern in model for pattern in self.reasoning_models)

    def budget_for(self, model: str | None) -> int | None:
        """The token cap this model gets: the reasoning budget, or the plain one."""
        if self.reasoning_max_tokens is not None and self.is_reasoning_model(model):
            return self.reasoning_max_tokens
        return self.max_tokens

    def call_kwargs(self, model: str | None = None, temperature: float | None = None) -> dict:
        """Keyword arguments for `model_openai.generate` / `generate_json`.

        `temperature` overrides the configured value — that is how the CLI
        flags stay authoritative over the file.
        """
        return {
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.budget_for(model),
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


class AgentConfig:
    def __init__(self, config_path: Path | str | None = None):
        configured = config_path or os.environ.get("LAYERED_VQA_AGENTS_CONFIG")
        # A path someone asked for must exist; the default one may not.
        self.required = configured is not None
        self.config_path = Path(configured or DEFAULT_CONFIG_PATH)
        self._lock = threading.Lock()
        self._config: dict | None = None
        self._cache: dict[tuple[str, str | None], AgentParams] = {}
        self._announced: set[tuple[str, str | None]] = set()

    @property
    def config(self) -> dict:
        with self._lock:
            if self._config is None:
                self._config = self._load()
            return self._config

    def _load(self) -> dict:
        if not self.config_path.exists():
            if self.required:
                raise FileNotFoundError(
                    f"Agent settings config not found at {self.config_path} "
                    "(from LAYERED_VQA_AGENTS_CONFIG)."
                )
            return BUILTIN
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self._validate(raw)
        return raw

    def _validate(self, raw: dict) -> None:
        """Reject unknown keys, so a typo fails loudly instead of running."""
        where = self.config_path.name
        unknown_top = set(raw) - {"agents", "reasoning_models"}
        if unknown_top:
            raise ValueError(f"{where}: unknown top-level key(s): {', '.join(sorted(unknown_top))}")

        agents = raw.get("agents") or {}
        unknown_agents = set(agents) - set(AGENTS)
        if unknown_agents:
            raise ValueError(
                f"{where}: unknown agent(s) {', '.join(sorted(unknown_agents))}; "
                f"known agents: {', '.join(AGENTS)}."
            )

        for agent, spec in agents.items():
            spec = spec or {}
            unknown = set(spec) - set(SETTINGS) - {"per_simulator"}
            if unknown:
                raise ValueError(
                    f"{where}: agent {agent!r} has unknown setting(s) {', '.join(sorted(unknown))}; "
                    f"known settings: {', '.join(SETTINGS)}."
                )
            for simulator, override in (spec.get("per_simulator") or {}).items():
                unknown = set(override or {}) - set(SETTINGS)
                if unknown:
                    raise ValueError(
                        f"{where}: {agent}.per_simulator.{simulator} has unknown setting(s) "
                        f"{', '.join(sorted(unknown))}; known settings: {', '.join(SETTINGS)}."
                    )

    @property
    def reasoning_models(self) -> tuple[str, ...]:
        configured = self.config.get("reasoning_models")
        # An explicit empty list disables the reasoning budget; an absent key
        # means the shipped list, like every other omitted setting.
        return tuple(BUILTIN["reasoning_models"] if configured is None else configured)

    def for_agent(self, agent: str, simulator: str | None = None) -> AgentParams:
        """Settings for one agent, optionally as overridden for one simulator.

        Three layers, each overriding the one before it: the shipped defaults
        for this agent, the agent's block in the file, and that block's
        `per_simulator` entry for this simulator. `per_simulator` is not
        layered — deleting a block from the file removes the override, which
        is how you give every simulator the same budget.
        """
        if agent not in AGENTS:
            raise ValueError(f"Unknown agent {agent!r}; known agents: {', '.join(AGENTS)}.")

        key = (agent, simulator)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        spec = dict((self.config.get("agents") or {}).get(agent) or {})
        overrides = (spec.pop("per_simulator", None) or {}).get(simulator) or {}
        settings = {}
        for layer in (BUILTIN["agents"][agent], spec, overrides):
            settings.update({k: v for k, v in layer.items() if k in SETTINGS})
        params = AgentParams(agent=agent, reasoning_models=self.reasoning_models, **settings)

        with self._lock:
            self._cache[key] = params
        self._announce(key, params)
        return params

    def _announce(self, key: tuple[str, str | None], params: AgentParams) -> None:
        """Say once, per agent, what its calls are being made with.

        Silent generation settings are worth spending a print on: temperature
        and token budget decide what a results table means, and neither is
        recoverable from the logs afterwards.
        """
        with self._lock:
            if key in self._announced:
                return
            self._announced.add(key)
        agent, simulator = key
        label = agent if simulator is None else f"{agent}/{simulator}"
        print(
            f"[agents] {label} -> temperature={params.temperature} "
            f"top_p={params.top_p} top_k={params.top_k} "
            f"max_tokens={params.max_tokens} timeout={params.timeout}s "
            f"retries={params.max_retries}"
        )

    # Named accessors: every call site goes through one of these three.
    def assistant(self, simulator: str | None = None) -> AgentParams:
        return self.for_agent("assistant", simulator)

    def user(self, simulator: str | None = None) -> AgentParams:
        return self.for_agent("user", simulator)

    def judge(self, simulator: str | None = None) -> AgentParams:
        return self.for_agent("judge", simulator)


agent_config = AgentConfig()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print the generation settings in force.")
    parser.add_argument("--model", type=str, default=None, help="Resolve token budgets for this model")
    args = parser.parse_args()

    print(f"[agents] config: {agent_config.config_path}")
    for agent_name in AGENTS:
        simulators = ("full", "sharded", "snowball", "recap") if agent_name == "assistant" else (None,)
        for sim in simulators:
            params = agent_config.for_agent(agent_name, sim)
            if args.model:
                print(f"          max_tokens for {args.model}: {params.budget_for(args.model)}")

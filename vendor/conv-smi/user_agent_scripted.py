"""User agents that reveal shards in a fixed, pre-computed order.

`UserAgent` lets an LLM choose which shard to reveal next. That is the right
behaviour when reveal order is incidental, but several Layered-VQA conditions
(EVIDENCE_FIRST_V, EVIDENCE_LAST_V, aligned vs. misaligned VQ) make reveal
order the independent variable, and an improvising user agent would destroy
the manipulation. `ScriptedUserAgent` walks `sample["shards"]` in the order
the exporter wrote them and costs nothing to run.

`shuffle=True` serves SHUFFLED_SHARDED_Q: the intent shard still opens the
conversation, and the remaining shards are revealed in a random order drawn
once per simulator (each run gets a fresh order; the realized order is
recoverable from the `shard_revealed` log entries in the trace).

The interface matches `UserAgent.generate_response` so the simulators can
swap between the agents without further changes.
"""

from __future__ import annotations

import random


class ScriptedUserAgent:
    def __init__(self, task, model: str | None = None, shuffle: bool = False):
        self.task = task
        self.task_name = task.get_task_name()
        # `model` is accepted so construction mirrors UserAgent, but a scripted
        # run never calls it. What the simulators log as `user_model` is this
        # attribute, read off whichever agent they built — the same shape as
        # `judge_model_of(task)`. Naming the assistant's model here instead
        # claimed a model chose the reveal order when the exporter did, and
        # made a scripted run indistinguishable from an LLM-driven one in the
        # record. The shuffled variant is named separately because it draws a
        # fresh order per run rather than replaying the exported one.
        self.model = "scripted-shuffle" if shuffle else "scripted"
        self.shuffle = shuffle
        self._order: list | None = None

    def _shard_order(self, sample) -> list:
        if self._order is None:
            shards = list(sample["shards"])
            if self.shuffle and len(shards) > 2:
                rest = shards[1:]
                random.shuffle(rest)
                shards = shards[:1] + rest
            self._order = shards
        return self._order

    def generate_response(self, conversation, sample, temperature: float | None = None):
        revealed = [
            msg["content"]["shard_id"]
            for msg in conversation
            if msg["role"] == "log"
            and msg["content"]["type"] in ("shard_revealed", "hint_revealed")
        ]

        for shard in self._shard_order(sample):
            if shard["shard_id"] not in revealed:
                return shard["shard"], shard["shard_id"]

        # Every shard is out; the simulator's own loop guard ends the run on
        # the next pass, so say nothing new.
        return "that's all I've got", -1


def shards_carry_images(samples) -> bool:
    return any(shard.get("images") for sample in samples for shard in sample.get("shards", []))


def resolve_user_agent(kind: str | None, samples) -> str:
    """Resolve a `--user_agent` choice against the dataset being run.

    Text-only datasets keep the historical default (`llm`). On datasets with
    per-shard images, reveal order is part of the experiment design, so the
    agent must be named explicitly: `scripted` delivers the exported order
    (the canonical / EVIDENCE_* / aligned / misaligned conditions), while an
    explicit `llm` is the improvised-order arm — a deliberate condition that
    logs under the llm conv_type (`sharded` / `snowball`), never something a
    forgotten flag should buy silently.
    """
    has_images = shards_carry_images(samples)
    if kind is None:
        if has_images:
            raise ValueError(
                "this dataset carries per-shard images, so reveal order is part of the "
                "design: pass --user_agent explicitly (scripted for the exported order, "
                "llm for an improvised order)."
            )
        return "llm"
    if kind == "llm" and has_images:
        print(
            "[warn] LLM user agent on per-shard images: reveal order is improvised, "
            "not the exported schedule — a different condition from scripted runs, "
            "logged under the llm conv_type."
        )
    return kind


def build_user_agent(kind: str, task, model: str):
    """Return the LLM-driven or scripted user agent by name."""
    if kind == "scripted":
        return ScriptedUserAgent(task, model)
    if kind == "scripted-shuffle":
        return ScriptedUserAgent(task, model, shuffle=True)
    if kind == "llm":
        from user_agent import UserAgent

        return UserAgent(task, model)
    raise ValueError(f"Unknown user agent kind: {kind}")

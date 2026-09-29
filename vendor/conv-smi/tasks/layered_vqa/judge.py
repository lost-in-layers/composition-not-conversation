"""Answer grading for open-ended visual questions.

Gold answers are free-form sentences ("A computer mouse.", "Brown, the cat is
at the bottom right corner"), so string equality is useless. A conversation is
graded once, when it ends, by handing its whole transcript to an LLM judge that
decides whether the assistant offered the reference answer at any point.

There is no classifier and no answer extractor in front of the judge: the
grading rules in `prompts/layered_vqa/layered_vqa_conversation_judge.txt` cover
what those two stages used to do, and doing it in one call removes two places
where a mislabelled turn could silently decide a run.

Every verdict is the LLM's: there is no exact-match shortcut, so identical
grading rules apply to a terse FULL reply and a rambling SHARDED conversation.
The one grading the model never sees is the empty transcript, which has
nothing to judge and is scored incorrect outright. Verdicts are cached on
disk, keyed by transcript; repeats are rare, but the cache makes a re-run
after a crash free.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

CONVERSATION_PROMPT_PATH = Path("prompts/layered_vqa/layered_vqa_conversation_judge.txt")
DEFAULT_CACHE_PATH = Path("logs/layered_vqa/judge_cache.jsonl")

_ARTICLES = {"a", "an", "the"}
_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    lowered = _PUNCTUATION.sub(" ", (text or "").lower())
    words = [word for word in _WHITESPACE.split(lowered) if word and word not in _ARTICLES]
    return " ".join(words)


class JudgeCache:
    """Append-only judgement cache, safe to share across simulator threads."""

    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._entries[record["key"]] = record

    def get(self, key: str) -> dict | None:
        with self._lock:
            return self._entries.get(key)

    def put(self, key: str, is_correct: bool, detail: dict | None = None) -> None:
        with self._lock:
            record = {"key": key, "is_correct": is_correct}
            if detail:
                record.update(detail)
            self._entries[key] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # errors="replace": a judge that emits an unpaired surrogate must
            # not be able to kill a run from inside the cache write, which sits
            # under a lock and behind every caller's error handling.
            with self.path.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(json.dumps(record) + "\n")


_SHARED_CACHES: dict[Path, JudgeCache] = {}
_SHARED_CACHES_LOCK = threading.Lock()


def shared_judge_cache(path: Path | str) -> JudgeCache:
    """One `JudgeCache` per cache file, shared process-wide.

    Every conversation builds its own task, and with it its own `AnswerJudge`
    — a private cache per judge would re-read the whole cache file once per
    conversation (quadratic over a launch) and make each instance's lock
    guard nothing. Keyed by resolved path, so tests pointing at private tmp
    files stay isolated from each other and from the real cache.
    """
    resolved = Path(path).expanduser().resolve()
    with _SHARED_CACHES_LOCK:
        cache = _SHARED_CACHES.get(resolved)
        if cache is None:
            cache = JudgeCache(resolved)
            _SHARED_CACHES[resolved] = cache
        return cache


def conversation_cache_key(task_id: str, transcript: str, reference: str, model: str) -> str:
    """Cache identity for one conversation-level judgement.

    Transcripts run to kilobytes, so the identity is hashed rather than stored
    inline. The judge model is part of it: a verdict from a different grader is
    a different measurement, and silently reusing it across models would make a
    grader swap invisible in the results. The `conv` prefix keeps these keys in
    their own namespace inside a shared cache file.
    """
    payload = "\x1f".join([model, task_id, normalize(transcript), normalize(reference)])
    return "conv:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AnswerJudge:
    def __init__(self, cache_path: Path | str | None = None, model: str | None = None):
        from agent_config import agent_config

        self.model = model or os.environ.get("LAYERED_VQA_JUDGE_MODEL", "gpt-4o-mini")
        self.params = agent_config.judge()
        self.cache = shared_judge_cache(
            cache_path
            or os.environ.get("LAYERED_VQA_JUDGE_CACHE", DEFAULT_CACHE_PATH)
        )
        self._prompt: str | None = None

    @property
    def prompt(self) -> str:
        if self._prompt is None:
            self._prompt = CONVERSATION_PROMPT_PATH.read_text(encoding="utf-8")
        return self._prompt

    def judge_conversation(
        self,
        transcript: str,
        reference: str,
        question: str,
        answer_description: str,
        task_id: str,
    ) -> dict:
        """Grade a finished conversation. `transcript` is its rendered text."""
        # The only verdict not from the LLM: an empty transcript has nothing
        # to grade, and prompting a judge with an empty conversation invites
        # a hallucinated one.
        if not transcript.strip():
            return {
                "score": 0.0,
                "is_correct": False,
                "grader": "deterministic",
                "answer_span": "",
            }

        key = conversation_cache_key(task_id, transcript, reference, self.model)
        cached = self.cache.get(key)
        if cached is not None:
            return {
                "score": 1.0 if cached["is_correct"] else 0.0,
                "is_correct": cached["is_correct"],
                "grader": "cache",
                "reasoning": cached.get("reasoning", ""),
                "answer_span": cached.get("answer_span", ""),
            }

        # Imported here so the module stays importable without credentials.
        from model_openai import generate_json

        populated = (
            self.prompt.replace("[[QUESTION]]", question)
            .replace("[[REFERENCE_ANSWER]]", reference)
            .replace("[[ANSWER_DESCRIPTION]]", answer_description)
            .replace("[[CONVERSATION]]", transcript)
        )
        try:
            response = generate_json(
                [{"role": "user", "content": populated}],
                model=self.model,
                return_metadata=True,
                **self.params.call_kwargs(self.model),
            )
            payload = response["message"]
            is_correct = bool(payload.get("is_correct", False))
            reasoning = payload.get("reasoning", "")
            answer_span = payload.get("answer_span", "") or ""
        except Exception as error:  # a judge outage must not kill the run
            return {
                "score": 0.0,
                "is_correct": False,
                "grader": "error",
                "answer_span": "",
                "error": f"{type(error).__name__}: {error}",
            }

        self.cache.put(
            key,
            is_correct,
            {
                "reasoning": reasoning,
                "answer_span": answer_span,
                "task_id": task_id,
                # The key is a hash, so keep something readable alongside it.
                "reference": reference,
            },
        )
        return {
            "score": 1.0 if is_correct else 0.0,
            "is_correct": is_correct,
            "grader": "llm",
            "reasoning": reasoning,
            "answer_span": answer_span,
        }

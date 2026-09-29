from datetime import datetime
import os

from multimodal import content_to_text

def load_env_vars(filepath='.env'):
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove 'export' if present and any leading/trailing whitespace
                if line.startswith('export '):
                    line = line[7:].strip()

                # Split on first '=' only
                if '=' in line:
                    key, value = line.split('=', 1)
                    # Remove any quotes around the value
                    value = value.strip('\'"')
                    os.environ[key.strip()] = value


def print_colored(text, color):
    if color == "red":
        print(f"\033[91m{text}\033[0m")
    elif color == "green":
        print(f"\033[92m{text}\033[0m")
    elif color == "blue":
        print(f"\033[94m{text}\033[0m")
    elif color == "purple":
        print(f"\033[95m{text}\033[0m")
    else:
        raise Exception(f"Unknown color: {color}")


def extract_conversation(simulation_trace, to_str=False, skip_system=False, only_last_turn=False, hide_image_paths=False):
    keep_roles = ["system", "assistant", "user"] if not skip_system else ["assistant", "user"]
    real_conversation = [msg for msg in simulation_trace if msg["role"] in keep_roles]
    if only_last_turn:
        # get all the user turns
        user_turn_idxs = [i for i, msg in enumerate(real_conversation) if msg["role"] == "user"]
        last_user_turn_idx = user_turn_idxs[-1]
        real_conversation = real_conversation[last_user_turn_idx + 1:]

    real_conversation = [{"role": msg["role"], "content": msg["content"]} for msg in real_conversation] # only keep role and content

    if to_str:
        # hide_image_paths renders image parts as a bare `[image]`. Required
        # for any string that lands in an agent prompt: image filenames
        # (`.../layer_2.png`) can name the answer.
        placeholder = "[image]" if hide_image_paths else None
        return "\n\n".join([f"[{msg['role']}] {content_to_text(msg['content'], image_placeholder=placeholder)}" for msg in real_conversation])
    else:
        return real_conversation

def date_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_judge_model_args(parser):
    """Register `--judge_model` and the retired `--system_model` alias."""
    parser.add_argument("--judge_model", type=str, default=None, help="Model that grades finished conversations (default: $LAYERED_VQA_JUDGE_MODEL, else gpt-4o-mini)")
    parser.add_argument("--system_model", type=str, default=None, help="Retired: the turn-classifying system agent is gone. Accepted and ignored so older commands still run; use --judge_model.")


def apply_judge_model(args):
    """Point the judge at `--judge_model`, before any task is constructed.

    `AnswerJudge.__init__` reads `LAYERED_VQA_JUDGE_MODEL` once, when the task
    builds it, so this has to run before the first `get_task()` call — which is
    also why the model travels as an env var rather than through four
    simulator constructors.
    """
    if getattr(args, "system_model", None):
        print("[warn] --system_model is retired and ignored; grading is done by --judge_model.")
    if getattr(args, "judge_model", None):
        os.environ["LAYERED_VQA_JUDGE_MODEL"] = args.judge_model


def judge_model_of(task):
    """The grader a task is configured with, for the log record."""
    return getattr(getattr(task, "judge", None), "model", "NA")


class JudgeUnavailableError(RuntimeError):
    """Grading failed (judge outage), so the run must not be logged.

    A logged run is a completed run to `get_run_counts`: writing it with a
    fallback zero would freeze the outage into the results and no launch
    would ever redo it. Aborting before the log write costs a re-run of the
    conversation, which is the resume design's normal price for a crash.
    """


def grade_conversation(task, trace, sample, verbose=False, graded_turns=None):
    """Grade a finished conversation and append its verdict to the trace.

    Every simulator ends the same way — one judge call, one `answer-evaluation`
    entry — so the four of them share this rather than each keeping its own
    copy, which is how their `is_correct` rules drifted apart in the first
    place. Mutates `trace`; returns `(is_correct, score)`.

    `graded_turns` narrows what the judge reads to a slice of `trace`; recap
    uses it to grade only its own exchange. The verdict is still logged onto
    the whole `trace`.

    Raises `JudgeUnavailableError` when the judge itself was unreachable, so
    callers never log a fallback zero as a completed run.
    """
    evaluation_return = task.evaluate_conversation(trace if graded_turns is None else graded_turns, sample)
    assert type(evaluation_return) is dict and ("score" in evaluation_return or "is_correct" in evaluation_return), "Evaluator function should return a dictionary with 'score' or 'is_correct' key"

    if evaluation_return.get("grader") == "error":
        # The conversation itself succeeded; only grading failed. Raise before
        # touching the trace so the caller's log write never happens and the
        # next launch re-runs this conversation.
        raise JudgeUnavailableError(f"judge failed to grade the conversation: {evaluation_return.get('error', 'unknown error')}")

    is_correct = evaluation_return.get("is_correct", None)
    score = evaluation_return.get("score", None)
    if score == 1.0 and not is_correct:
        is_correct = True

    # `exact_answer` is the assistant's own words that gave the answer, quoted
    # back by the judge. The key name predates the extractor's removal; the
    # viewer reads it.
    answer_span = evaluation_return.get("answer_span", "")
    trace.append({"role": "log", "content": {"type": "answer-evaluation", "exact_answer": answer_span, "is_correct": is_correct, "score": score, "evaluation_return": evaluation_return}, "timestamp": date_str()})
    if verbose:
        print_colored(f"[log] answer evaluation:\n```{answer_span}\n```\n({'correct' if is_correct else 'incorrect'}; score: {score})", "blue")

    if is_correct:
        # No longer means "stopped early" — conversations always run to shard
        # exhaustion now. It marks that the conversation was solved.
        trace.append({"role": "log", "content": {"type": "conversation-completed", "is_correct": is_correct}, "timestamp": date_str()})
        if verbose:
            print_colored(f"[log] conversation completed: {is_correct}; score: {score}", "blue")

    return is_correct, score

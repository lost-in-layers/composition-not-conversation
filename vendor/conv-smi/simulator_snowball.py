import random
import tqdm

from agent_config import agent_config
from utils import add_judge_model_args, apply_judge_model, print_colored, extract_conversation, date_str, grade_conversation, judge_model_of
from utils_log import log_conversation, temperature_suffix
from user_agent_scripted import build_user_agent, resolve_user_agent
from tasks import get_task
from model_openai import generate
from concurrent.futures import ThreadPoolExecutor
from utils_log import get_run_counts
from collections import Counter
from multimodal import content_to_text, describe_images, image_base_dir, multimodal_content, sample_images

snowball_message = """Just to reiterate,
{HINTS_SO_FAR}

Also,

{LAST_HINT}
"""


def snowball_conv_type(user_agent, temperature=None):
    """Must match between logging and `get_run_counts`, or completed runs are
    never counted and every launch redoes them.

    `temperature` defaults to the configured assistant temperature, so a
    non-baseline value in `agents.yaml` separates those runs from the 1.0 ones
    instead of quietly mixing with them.
    """
    if temperature is None:
        temperature = agent_config.assistant("snowball").temperature
    base = f"snowball{temperature_suffix(temperature)}"
    return base if user_agent == "llm" else f"{base}-{user_agent}"


class ConversationSimulatorSnowball:
    def __init__(self, task_name, sample, assistant_model="gpt-4o-mini", user_model="gpt-4o-mini", user_agent="llm", log_folder="logs"): # max_turns=10
        assert task_name in ["layered_vqa"]

        self.task_name = task_name
        self.task = get_task(task_name)
        self.dataset_fn = self.task.get_dataset_file()
        self.sample = sample
        self.assistant_model = assistant_model
        self.user_model = user_model
        self.user_agent_kind = user_agent
        self.user_agent = build_user_agent(user_agent, self.task, user_model)
        self.assistant_params = agent_config.assistant("snowball")
        # Scripted runs answer a different question than improvised ones; keep
        # their logs from being counted as the same runs.
        self.conv_type = snowball_conv_type(user_agent, self.assistant_params.temperature)
        self.log_folder = log_folder

        self.system_message = self.task.generate_system_prompt(self.sample)
        self.user_response_template = snowball_message

        self.trace = [{"role": "system", "content": self.system_message, "timestamp": date_str()}]
        self.image_base_dir = image_base_dir(self.dataset_fn)

    def get_num_turns(self, participant="assistant"):
        return sum(1 for msg in self.trace if msg["role"] == participant)

    def run(self, verbose=False, save_log=True):

        assistant_kwargs = self.assistant_params.call_kwargs(self.assistant_model)
        user_response_history = []

        shards = self.sample["shards"]

        # Runs to hint exhaustion: grading happens once at the end, over the
        # whole trace, so there is nothing to stop early for. The cap only
        # catches a stuck LLM user agent that keeps revealing nothing.
        max_user_turns = 2 * len(shards) + 6

        while True:
            revealed_shard_ids = set([msg["content"]["shard_id"] for msg in self.trace if msg["role"] == "log" and msg["content"]["type"] == "hint_revealed"])
            all_shards_revealed = len(revealed_shard_ids) >= len(shards)
            if all_shards_revealed:
                if verbose:
                    print_colored(f"[log] all hints revealed ({revealed_shard_ids} / {len(shards)})", "blue")
                break # no need to keep going, nothing else to reveal
            if self.get_num_turns("user") >= max_user_turns:
                # Abort unlogged: an under-revealed conversation would be a
                # different condition, so it must not land in the results.
                raise RuntimeError(f"aborting {self.sample['task_id']}: {max_user_turns} user turns without revealing every hint ({sorted(revealed_shard_ids)} of {len(shards)}); the user agent is stuck")

            # 1. get a user response
            user_response, hint_revealed_id = self.user_agent.generate_response(self.trace, self.sample)

            # Sample-level images arrive once, on the first user turn.
            turn_images = sample_images(
                self.sample, include_sample=self.get_num_turns("user") == 0
            )

            if hint_revealed_id != -1:
                self.trace.append({"role": "log", "content": {"type": "hint_revealed", "shard_id": hint_revealed_id}, "timestamp": date_str()})
                user_response_history.append(user_response)
                if verbose:
                    print_colored(f"[log] hint revealed: {hint_revealed_id}", "blue")

                if len(revealed_shard_ids) > 0:
                    # Create a new user response that includes all hints so far
                    user_response = self.user_response_template.format(HINTS_SO_FAR="\n".join([f"- {hint}" for hint in user_response_history[:-1]]), LAST_HINT=user_response_history[-1])

                # A snowball turn explicitly repeats everything so far, so it
                # has to re-attach every image already revealed, not just the
                # newest: the repeated hint text for an image turn ("here's
                # another part of the image") carries nothing without its
                # image.
                revealed_in_order = [msg["content"]["shard_id"] for msg in self.trace if msg["role"] == "log" and msg["content"]["type"] == "hint_revealed"]
                for revealed_id in revealed_in_order:
                    turn_images += sample_images(self.sample, shard_id=revealed_id, include_sample=False)
            # A turn that reveals nothing (the LLM agent answering a
            # clarifying question) goes out as the agent wrote it: replacing
            # it with a re-iteration of the previous hint would discard what
            # the agent actually said, and there is nothing new to re-attach.

            image_caption = describe_images(turn_images)
            if image_caption:
                user_response = f"{user_response}\n{image_caption}"
            user_response = multimodal_content(user_response, turn_images)
            self.trace.append({"role": "user", "content": user_response, "timestamp": date_str()})
            if verbose:
                print_colored(f"[user] {content_to_text(user_response)}", "green")

            # 2. get the assistant's response
            assistant_response_obj = generate(extract_conversation(self.trace, to_str=False), model=self.assistant_model, return_metadata=True, image_base_dir=self.image_base_dir, **assistant_kwargs)
            assistant_response = assistant_response_obj["message"]
            self.trace.append({"role": "assistant", "content": assistant_response, "timestamp": date_str()})
            if verbose:
                print_colored(f"[assistant] {assistant_response}", "red")

        # 3. grade the finished conversation, once
        is_correct, score = grade_conversation(self.task, self.trace, self.sample, verbose=verbose)

        if save_log:
            log_conversation(self.conv_type, self.task.get_task_name(), self.sample["task_id"], self.dataset_fn, self.assistant_model, self.user_agent.model, self.trace, is_correct, score, judge_model=judge_model_of(self.task), log_folder=self.log_folder)
        return is_correct, score


def single_run(todo):
    # Catch per run: an uncaught exception inside executor.map would abort
    # the whole launch when its result is consumed, discarding the queue.
    try:
        conversation_simulator = ConversationSimulatorSnowball(args.task, todo["sample"], assistant_model=todo["assistant_model"], user_model=todo["user_model"], user_agent=todo["user_agent"], log_folder=args.log_folder)
        conversation_simulator.run(verbose=args.verbose, save_log=True)
    except Exception:
        import traceback
        tqdm.tqdm.write(f"\033[91m[Error on {todo['sample']['task_id']}; {todo['assistant_model']}; {todo['conv_type']}]:\n{traceback.format_exc()}\033[0m")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="layered_vqa", choices=["layered_vqa"])
    parser.add_argument("--assistant_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--user_agent", type=str, default=None, choices=["llm", "scripted"], help="'llm' lets the user model choose the next shard; 'scripted' reveals shards in dataset order. Defaults to 'llm' on text-only datasets; datasets with per-shard images require an explicit choice, because reveal order is part of the design")
    parser.add_argument("--log_folder", type=str, default="logs", help="Log folder to use")
    parser.add_argument("--N_snowball_runs", type=int, default=2, help="Number of snowball runs per model")
    parser.add_argument("--N_workers", type=int, default=7, help="Number of workers to run experiments with")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--models", nargs="+", default=["local-vlm"],
                        help="Models to run experiments with; names come from models.yaml")
    add_judge_model_args(parser)
    args = parser.parse_args()
    apply_judge_model(args)

    task = get_task(args.task)
    samples = task.get_samples()
    print(f"Loaded {len(samples)} samples ({args.task})")
    dataset_fn = task.get_dataset_file()
    try:
        # Fail (or warn, for an explicit llm choice) before any todo is paid for.
        args.user_agent = resolve_user_agent(args.user_agent, samples)
    except ValueError as e:
        parser.error(str(e))
    random.shuffle(samples)
    all_todos = []

    conv_type = snowball_conv_type(args.user_agent)
    for assistant_model in args.models:
        run_counts = get_run_counts(conv_type, args.task, assistant_model, dataset_fn, log_folder=args.log_folder)

        for sample in samples:
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": conv_type, "user_model": args.user_model, "user_agent": args.user_agent}] * (args.N_snowball_runs - run_counts[sample["task_id"]])

    random.shuffle(all_todos)

    print(f"Running {len(all_todos)} conversations")
    print(Counter([todo["assistant_model"] for todo in all_todos]))
    print(Counter([todo["conv_type"] for todo in all_todos]))

    with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
        list(tqdm.tqdm(executor.map(single_run, all_todos), total=len(all_todos)))

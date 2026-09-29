import argparse
import random
import multiprocessing
import json
import tqdm
from agent_config import agent_config
from simulator_sharded import ConversationSimulatorSharded
from simulator_full import ConversationSimulatorFull
from concurrent.futures import ThreadPoolExecutor
from utils import add_judge_model_args, apply_judge_model
from user_agent_scripted import resolve_user_agent
from utils_log import get_run_counts, temperature_suffix
from collections import Counter

def run_simulation(todo):
    dataset_fn = todo["dataset_fn"]
    try:
        # None lets each simulator take its temperature from agents.yaml; the
        # todo carries a number only when a --*_temperature flag overrode it.
        assistant_temp = todo.get("assistant_temperature")
        user_temp = todo.get("user_temperature")
        if todo["conv_type"].startswith("full"):
            conversation_simulator = ConversationSimulatorFull(todo["sample"], assistant_model=todo["assistant_model"], temperature=assistant_temp, dataset_fn=dataset_fn, log_folder=args.log_folder)
        elif todo["conv_type"].startswith("shuffle-concat"):
            conversation_simulator = ConversationSimulatorFull(todo["sample"], assistant_model=todo["assistant_model"], run_shuffle_concat=True, temperature=assistant_temp, dataset_fn=dataset_fn, log_folder=args.log_folder)
        elif todo["conv_type"].startswith("concat"):
            conversation_simulator = ConversationSimulatorFull(todo["sample"], assistant_model=todo["assistant_model"], run_concat=True, temperature=assistant_temp, dataset_fn=dataset_fn, log_folder=args.log_folder)
        elif todo["conv_type"].startswith("sharded"):
            conversation_simulator = ConversationSimulatorSharded(todo["sample"], assistant_model=todo["assistant_model"], user_model=todo["user_model"], assistant_temperature=assistant_temp, user_temperature=user_temp, dataset_fn=dataset_fn, log_folder=args.log_folder, user_agent=args.user_agent)
        conversation_simulator.run(verbose=args.verbose)

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        tqdm.tqdm.write(f"\033[91m [Error on {todo['sample']['task_id']}; {todo['assistant_model']}; {todo['conv_type']}]:\n{error_msg}\033[0m")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_file", type=str, default="../../data/derived/lic/layered_vqa_text.json", help="Dataset file to use")

    parser.add_argument("--N_full_runs", type=int, default=1, help="Number of full runs per model")
    parser.add_argument("--N_concat_runs", type=int, default=1, help="Number of concat runs per model")
    parser.add_argument("--N_shuffle_concat_runs", type=int, default=0, help="Number of shuffle-concat runs per model (intent shard first, remaining shards in random order, one turn)")
    parser.add_argument("--N_sharded_runs", type=int, default=1, help="Number of sharded runs per model")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"], # `, "gpt-4o"`
                        help="List of models to run experiments with")
    parser.add_argument("--tasks", nargs="+", default=["layered_vqa"], help="Tasks to run experiments with")
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini", help="User model to use")
    add_judge_model_args(parser)
    parser.add_argument("--user_agent", type=str, default=None, choices=["llm", "scripted", "scripted-shuffle"], help="'llm' lets the user model choose the next shard; 'scripted' reveals shards in dataset order, for conditions where reveal order is the manipulation; 'scripted-shuffle' keeps the intent shard first and reveals the rest in random order. Defaults to 'llm' on text-only datasets; datasets with per-shard images require an explicit choice, because reveal order is part of the design")
    parser.add_argument("--N_workers", type=int, default=1, help="Number of workers to run experiments with")
    parser.add_argument("--log_folder", type=str, default="logs", help="Log folder to use")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    parser.add_argument("--assistant_temperature", type=float, default=None, help="Temperature to use for assistant models (default: agents.yaml)")
    parser.add_argument("--user_temperature", type=float, default=None, help="Temperature to use for user models (default: agents.yaml)")

    args = parser.parse_args()
    apply_judge_model(args)

    # windows fix dataset_file to be unix format
    dataset_fn = args.dataset_file
    if args.dataset_file.startswith(".\\"):
        dataset_fn = args.dataset_file[2:]
    dataset_fn = dataset_fn.replace("\\", "/")

    with open(dataset_fn, "r") as f:
        samples = json.load(f)

    samples = [d for d in samples if d["task"] in args.tasks]

    if args.N_sharded_runs > 0:
        try:
            # Fail (or warn, for an explicit llm choice) before any todo is
            # built or paid for.
            args.user_agent = resolve_user_agent(args.user_agent, samples)
        except ValueError as e:
            parser.error(str(e))
    elif args.user_agent is None:
        args.user_agent = "llm"  # unused without sharded runs; keep conv_type strings well-formed

    print(f"Loaded {len(samples)} samples")
    random.shuffle(samples)
    all_todos = []

    # The temperatures the simulators will actually run at: the flag when it
    # was passed, else this simulator's value from agents.yaml. Resolved per
    # simulator because agents.yaml can override the assistant's temperature
    # for one of them, and a single number here would then name the wrong
    # conv_type for the other.
    def assistant_temp(simulator):
        return agent_config.assistant(simulator).temperature if args.assistant_temperature is None else args.assistant_temperature

    user_temp = agent_config.user().temperature if args.user_temperature is None else args.user_temperature

    sharded_extra = temperature_suffix(assistant_temp("sharded"), user_temp)
    st_extra = temperature_suffix(assistant_temp("full"))
    # Must match the conv_type ConversationSimulatorSharded writes, or already
    # completed runs are never counted and every launch redoes them.
    agent_extra = "" if args.user_agent == "llm" else f"-{args.user_agent}"
    sharded_ct, full_ct, concat_ct, shuffle_concat_ct = f"sharded{sharded_extra}{agent_extra}", f"full{st_extra}", f"concat{st_extra}", f"shuffle-concat{st_extra}"

    all_tasks = list(set([sample["task"] for sample in samples]))
    for assistant_model in args.models:
        sharded_run_counts, full_run_counts, concat_run_counts, shuffle_concat_run_counts = Counter(), Counter(), Counter(), Counter()
        for task in all_tasks:
            sharded_run_counts.update(get_run_counts(sharded_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
            full_run_counts.update(get_run_counts(full_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
            concat_run_counts.update(get_run_counts(concat_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
            shuffle_concat_run_counts.update(get_run_counts(shuffle_concat_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
        print(f"Sharded run counts: {sharded_run_counts}")
        print(f"Full run counts: {full_run_counts}")
        print(f"Concat run counts: {concat_run_counts}")
        print(f"Shuffle-concat run counts: {shuffle_concat_run_counts}")

        for sample in samples:
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": full_ct, "dataset_fn": dataset_fn}] * (args.N_full_runs - full_run_counts[sample["task_id"]])
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": concat_ct, "dataset_fn": dataset_fn}] * (args.N_concat_runs - concat_run_counts[sample["task_id"]])
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": shuffle_concat_ct, "dataset_fn": dataset_fn}] * (args.N_shuffle_concat_runs - shuffle_concat_run_counts[sample["task_id"]])
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": sharded_ct, "user_model": args.user_model, "dataset_fn": dataset_fn}] * (args.N_sharded_runs - sharded_run_counts[sample["task_id"]])

    # Only flag-supplied temperatures travel on the todo; a todo without them
    # lets each simulator read agents.yaml itself.
    for todo in all_todos:
        if args.assistant_temperature is not None:
            todo["assistant_temperature"] = args.assistant_temperature
        if args.user_temperature is not None:
            todo["user_temperature"] = args.user_temperature

    random.shuffle(all_todos)

    print(f"Running {len(all_todos)} conversations")
    print(Counter([todo["assistant_model"] for todo in all_todos]))
    print(Counter([todo["conv_type"] for todo in all_todos]))

    with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
        list(tqdm.tqdm(executor.map(run_simulation, all_todos), total=len(all_todos)))

import os
import json
import argparse
import tqdm
from concurrent.futures import ThreadPoolExecutor
from agent_config import agent_config
from utils import add_judge_model_args, apply_judge_model, extract_conversation, date_str, grade_conversation, judge_model_of
from utils_log import log_conversation, temperature_suffix
from model_openai import generate
from tasks import get_task
from multimodal import all_sample_images, image_base_dir, multimodal_content

recap_message = """Just to recapitulate, the entire task is:

[[SAMPLE_TEXT]]

Now please complete the task correctly taking all the information into account."""


def recap_conv_type(recap_kind, temperature=None):
    """Must match between logging and the already-completed scan below, or
    finished recaps are replayed (and re-billed) on every launch.

    `temperature` defaults to the configured assistant temperature, so a
    non-baseline value in `agents.yaml` separates those runs from the 1.0 ones.
    """
    if temperature is None:
        temperature = agent_config.assistant("recap").temperature
    return f"recap-{recap_kind}{temperature_suffix(temperature)}"


class RecapSimulator:
    def __init__(self, task_name, log_folder="logs"):
        self.task_name = task_name
        self.task = get_task(task_name)
        self.dataset_fn = self.task.get_dataset_file()
        self.log_folder = log_folder
        self.assistant_params = agent_config.assistant("recap")

    def run_recap_sample(self, sharded_log, conv_type, save_log=True):
        sharded_conv_id = sharded_log["conv_id"]

        sharded_trace = sharded_log["trace"]
        sharded_sample = self.task.get_sample(sharded_log["task_id"])

        assistant_model, user_model = sharded_log["assistant_model"], sharded_log["user_model"]

        recap_trace = sharded_trace.copy()

        # 1. prep the user recap message
        # The prompt names the images it expects to follow, so the turn has to
        # carry the same image set or the recap announces attachments that are
        # not there. Tasks that want a recap-specific set (layered_vqa re-sends
        # the full RGB, not the per-shard layers) provide populate_recap_prompt.
        # startswith, not ==: conv_type carries a temperature suffix when the
        # configured assistant temperature is not the 1.0 baseline.
        is_recap_full = conv_type.startswith("recap-full")
        if not is_recap_full and hasattr(self.task, "populate_recap_prompt"):
            sample_text, recap_images = self.task.populate_recap_prompt(sharded_sample)
        else:
            sample_text = self.task.populate_fully_specific_prompt(sharded_sample) if is_recap_full else self.task.populate_concat_prompt(sharded_sample)
            recap_images = all_sample_images(sharded_sample)
        recap_message_populated = recap_message.replace("[[SAMPLE_TEXT]]", sample_text)
        recap_content = multimodal_content(recap_message_populated, recap_images)
        recap_trace.append({"role": "user", "content": recap_content, "timestamp": date_str()})

        assistant_conversation = extract_conversation(recap_trace, to_str=False, skip_system=False)

        # 2. get the assistant's response
        recap_response_obj = generate(assistant_conversation, model=assistant_model, return_metadata=True, image_base_dir=image_base_dir(self.dataset_fn), **self.assistant_params.call_kwargs(assistant_model))
        recap_response = recap_response_obj["message"]
        recap_trace.append({"role": "assistant", "content": recap_response, "timestamp": date_str()})

        # 3. grade the recap exchange only — the recap turn and the reply to
        # it — never the replayed sharded trace it is appended to. RECAP asks
        # whether re-stating everything recovers the answer; the judge credits
        # an answer given at any point in what it is shown, and sharded runs no
        # longer stop when they answer correctly, so grading the full trace
        # would score the recap for the original conversation's work.
        is_correct, score = grade_conversation(self.task, recap_trace, sharded_sample, graded_turns=recap_trace[-2:])

        if save_log:
            additional_info = {"source_conv_id": sharded_conv_id, "source_conv_type": sharded_log.get("conv_type"), "source_is_correct": sharded_log["is_correct"]}
            log_conversation(conv_type, self.task.get_task_name(), sharded_sample["task_id"], self.dataset_fn, assistant_model, user_model, recap_trace, is_correct=is_correct, score=score, judge_model=judge_model_of(self.task), additional_info=additional_info, log_folder=self.log_folder)

        return is_correct, score

def iter_logs(folder):
    """Yield every record in every .jsonl file of a log folder, if it exists."""
    if not os.path.isdir(folder):
        return
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(folder, fn), "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def passes_model_filter(record, models):
    model_name = record["assistant_model"][2:] if record["assistant_model"].startswith("t-") else record["assistant_model"]
    return models is None or model_name in models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="layered_vqa")
    parser.add_argument("--models", type=str, nargs="+")
    parser.add_argument("--N_workers", type=int, default=5)
    parser.add_argument("--log_folder", type=str, default="logs", help="Log folder to use")
    parser.add_argument("--sharded_conv_types", type=str, nargs="+", default=["sharded", "sharded-scripted"],
                        help="Which sharded log folders to replay (scripted runs log separately)")
    add_judge_model_args(parser)
    args = parser.parse_args()
    apply_judge_model(args)

    task_name = args.task
    todo_convs = []

    task = get_task(task_name)
    task_ids = set([d["task_id"] for d in task.get_samples()])
    # Only replay sharded runs of the dataset this launch points at (for
    # layered_vqa that is LAYERED_VQA_DATASET); logs store the file's basename.
    dataset_basename = os.path.basename(task.get_dataset_file())

    for recap_type in ["concat"]:
        conv_type = recap_conv_type(recap_type)

        conv_id2log = {}
        for source_conv_type in args.sharded_conv_types:
            for log in iter_logs(f"{args.log_folder}/{task_name}/{source_conv_type}"):
                if log["dataset_fn"] != dataset_basename or log["task_id"] not in task_ids:
                    continue
                if passes_model_filter(log, args.models):
                    conv_id2log[log["conv_id"]] = log

        already_completed_conv_ids = set([])
        for conv in iter_logs(f"{args.log_folder}/{task_name}/{conv_type}"):
            if conv["dataset_fn"] != dataset_basename or conv["task_id"] not in task_ids:
                continue
            if passes_model_filter(conv, args.models):
                already_completed_conv_ids.add(conv["source_conv_id"])

        for conv_id, log in conv_id2log.items():
            if conv_id not in already_completed_conv_ids:
                todo_convs.append({"conv": log, "conv_type": conv_type})

        print(f"[{conv_type}] Found {len(conv_id2log)} sharded conversations ({dataset_basename}), {len(already_completed_conv_ids)} already completed; todo: {len(todo_convs)}")

    recap_simulator = RecapSimulator(task_name=task_name, log_folder=args.log_folder)

    def run_one(todo):
        # Catch per run: an uncaught exception inside executor.map would
        # abort the whole launch when its result is consumed, discarding the
        # queue. Failed recaps stay unlogged and are retried next launch.
        try:
            recap_simulator.run_recap_sample(todo["conv"], todo["conv_type"])
        except Exception:
            import traceback
            tqdm.tqdm.write(f"\033[91m[Error on {todo['conv']['task_id']}; {todo['conv']['assistant_model']}; {todo['conv_type']}]:\n{traceback.format_exc()}\033[0m")

    with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
        list(tqdm.tqdm(executor.map(run_one, todo_convs), total=len(todo_convs)))

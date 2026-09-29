import random

from agent_config import agent_config
from utils_log import log_conversation, temperature_suffix
from model_openai import generate
from tasks import get_task
from utils import add_judge_model_args, apply_judge_model, grade_conversation, judge_model_of
from multimodal import all_sample_images, content_to_text, image_base_dir, multimodal_content


class ConversationSimulatorFull:
    def __init__(self, sample, assistant_model, run_concat=False, run_shuffle_concat=False, temperature=None, dataset_fn=None, log_folder=None):
        self.task_name = sample["task"]
        self.task = get_task(self.task_name)
        self.dataset_fn = dataset_fn
        self.sample = sample
        self.assistant_model = assistant_model
        self.run_concat = run_concat
        self.run_shuffle_concat = run_shuffle_concat
        self.log_folder = log_folder
        # temperature=None means "whatever agents.yaml says"; an explicit value
        # (the --assistant_temperature flag) overrides it.
        self.assistant_params = agent_config.assistant("full")
        self.temperature = self.assistant_params.temperature if temperature is None else temperature

        self.system_message = self.task.generate_system_prompt(self.sample)
        self.image_base_dir = image_base_dir(self.dataset_fn)

    def run(self, verbose=False, save_log=True):
        if self.run_shuffle_concat and self.run_concat:
            raise ValueError("Cannot set both run_concat and run_shuffle_concat to True")

        sample = self.sample
        if self.run_shuffle_concat:
            conv_type = "shuffle-concat"

            # Shuffle only the non-intent shards: the intent shard stays first
            # so the task is still announced before its randomized details.
            # Work on a copy — the sample dict is shared with other queued
            # runs (full/concat/sharded) of the same sample, and shuffling it
            # in place would silently reorder their shards too.
            shards = list(sample["shards"])
            rest = shards[1:]
            random.shuffle(rest)
            sample = {**sample, "shards": shards[:1] + rest}

            input_prompt = self.task.populate_concat_prompt(sample)
        elif self.run_concat:
            conv_type = "concat"
            input_prompt = self.task.populate_concat_prompt(sample)
        else:
            conv_type = "full"
            input_prompt = self.task.populate_fully_specific_prompt(sample)

        input_prompt = multimodal_content(input_prompt, all_sample_images(sample))

        if verbose:
            print(f"\033[92m[user] {content_to_text(input_prompt)}\033[0m")

        # custom output dir for different temperatures to not mix up
        conv_type = f"{conv_type}{temperature_suffix(self.temperature)}"

        assistant_response_obj = generate(
            [{"role": "system", "content": self.system_message},{"role": "user", "content": input_prompt}],
            model=self.assistant_model,
            return_metadata=True,
            image_base_dir=self.image_base_dir,
            **self.assistant_params.call_kwargs(self.assistant_model, temperature=self.temperature),
        )

        assistant_response = assistant_response_obj["message"]
        if verbose:
            print(f"\033[91m[assistant] {assistant_response}\033[0m")

        trace = [{"role": "system", "content": self.system_message}, {"role": "user", "content": input_prompt}, {"role": "assistant", "content": assistant_response}]

        is_correct, score = grade_conversation(self.task, trace, self.sample, verbose=verbose)

        if save_log:
            log_conversation(conv_type, self.task_name, self.sample["task_id"], self.dataset_fn, assistant_model=self.assistant_model, user_model="NA", trace=trace, is_correct=is_correct, score=score, judge_model=judge_model_of(self.task), log_folder=self.log_folder)
        return is_correct, score



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--task", type=str, default="layered_vqa")
    parser.add_argument("--run_concat", action="store_true")
    parser.add_argument("--run_shuffle_concat", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    add_judge_model_args(parser)
    args = parser.parse_args()
    apply_judge_model(args)

    if args.run_concat and args.run_shuffle_concat:
        raise ValueError("Cannot set both run_concat and run_shuffle_concat to True")

    task = get_task(args.task)
    sample = random.choice(task.get_samples())

    # dataset_fn anchors relative image paths; without it they resolve
    # against the CWD and every image-carrying sample fails to load.
    conversation_simulator = ConversationSimulatorFull(sample, args.assistant_model, args.run_concat, args.run_shuffle_concat, dataset_fn=task.get_dataset_file())
    conversation_simulator.run(verbose=args.verbose, save_log=False)

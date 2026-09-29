import random

from agent_config import agent_config
from utils import add_judge_model_args, apply_judge_model, print_colored, extract_conversation, date_str, grade_conversation, judge_model_of
from utils_log import log_conversation, temperature_suffix
from model_openai import generate
from user_agent_scripted import build_user_agent, resolve_user_agent
from tasks import get_task
from multimodal import content_to_text, describe_images, image_base_dir, multimodal_content, sample_images


class ConversationSimulatorSharded:
    def __init__(self, sample, assistant_model="gpt-4o-mini", user_model="gpt-4o-mini", assistant_temperature=None, user_temperature=None, dataset_fn=None, log_folder="logs", user_agent="llm"):
        self.task_name = sample["task"]
        self.task = get_task(self.task_name)
        self.dataset_fn = dataset_fn
        self.sample = sample
        self.assistant_model = assistant_model
        self.user_model = user_model
        self.user_agent_kind = user_agent
        self.user_agent = build_user_agent(user_agent, self.task, user_model)
        self.log_folder = log_folder
        self.system_message = self.task.generate_system_prompt(self.sample)

        # None means "whatever agents.yaml says"; an explicit value (the
        # --assistant_temperature / --user_temperature flags) overrides it.
        self.assistant_params = agent_config.assistant("sharded")
        self.assistant_temperature = self.assistant_params.temperature if assistant_temperature is None else assistant_temperature
        self.user_temperature = agent_config.user().temperature if user_temperature is None else user_temperature

        self.trace = [{"role": "system", "content": self.system_message, "timestamp": date_str()}]
        self.image_base_dir = image_base_dir(self.dataset_fn)

    def get_num_turns(self, participant="assistant"):
        return sum(1 for msg in self.trace if msg["role"] == participant)

    def run(self, verbose=False, save_log=True):

        assistant_kwargs = self.assistant_params.call_kwargs(self.assistant_model, temperature=self.assistant_temperature)

        shards = self.sample["shards"]

        # The conversation always runs to shard exhaustion: grading happens
        # once at the end, over the whole trace, so there is nothing to stop
        # early for. The cap only catches a stuck LLM user agent (one that
        # keeps revealing nothing); a healthy run reveals a shard most turns
        # and never gets near it.
        max_user_turns = 2 * len(shards) + 6

        while True:
            revealed_shard_ids = set([msg["content"]["shard_id"] for msg in self.trace if msg["role"] == "log" and msg["content"]["type"] == "shard_revealed"])
            all_shards_revealed = len(revealed_shard_ids) >= len(shards)
            if all_shards_revealed:
                if verbose:
                    print_colored(f"[log] all shards revealed ({revealed_shard_ids} / {len(shards)})", "blue")
                break # no need to keep going, nothing else to reveal
            if self.get_num_turns("user") >= max_user_turns:
                # Abort unlogged: an under-revealed conversation would be a
                # different condition, so it must not land in the results.
                raise RuntimeError(f"aborting {self.sample['task_id']}: {max_user_turns} user turns without revealing every shard ({sorted(revealed_shard_ids)} of {len(shards)}); the user agent is stuck")

            # 1. get a user response
            user_response, shard_revealed_id = self.user_agent.generate_response(self.trace, self.sample, temperature=self.user_temperature)
            turn_images = sample_images(
                self.sample,
                shard_id=shard_revealed_id,
                include_sample=self.get_num_turns("user") == 0,
            )
            image_caption = describe_images(turn_images)
            if image_caption:
                user_response = f"{user_response}\n{image_caption}"
            user_response = multimodal_content(user_response, turn_images)
            self.trace.append({"role": "user", "content": user_response, "timestamp": date_str()})
            if verbose:
                print_colored(f"[user] {content_to_text(user_response)}", "green")

            if shard_revealed_id != -1:
                self.trace.append({"role": "log", "content": {"type": "shard_revealed", "shard_id": shard_revealed_id}, "timestamp": date_str()})
                if verbose:
                    print_colored(f"[log] shard revealed: {shard_revealed_id}", "blue")

            # 2. get the assistant's response
            assistant_response_obj = generate(extract_conversation(self.trace, to_str=False), model=self.assistant_model, return_metadata=True, image_base_dir=self.image_base_dir, **assistant_kwargs)
            assistant_response = assistant_response_obj["message"]
            self.trace.append({"role": "assistant", "content": assistant_response, "timestamp": date_str()})
            if verbose:
                print_colored(f"[assistant] {assistant_response}", "red")

        # 3. grade the finished conversation, once
        is_correct, score = grade_conversation(self.task, self.trace, self.sample, verbose=verbose)

        if save_log:
            conv_type = f"sharded{temperature_suffix(self.assistant_temperature, self.user_temperature)}"
            if self.user_agent_kind != "llm":
                # Scripted runs answer a different question than improvised
                # ones; keep their logs from being counted as the same runs.
                conv_type = f"{conv_type}-{self.user_agent_kind}"
            log_conversation(conv_type, self.task.get_task_name(), self.sample["task_id"], self.dataset_fn, self.assistant_model, self.user_agent.model, self.trace, is_correct, score, judge_model=judge_model_of(self.task), log_folder=self.log_folder)
        return is_correct, score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="layered_vqa")
    parser.add_argument("--assistant_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--user_agent", type=str, default=None, choices=["llm", "scripted", "scripted-shuffle"], help="Defaults to 'llm' on text-only datasets; datasets with per-shard images require an explicit choice")
    add_judge_model_args(parser)
    args = parser.parse_args()
    apply_judge_model(args)

    task = get_task(args.task)
    dataset_fn = task.get_dataset_file()
    sample = random.choice(task.get_samples())
    try:
        user_agent = resolve_user_agent(args.user_agent, [sample])
    except ValueError as e:
        parser.error(str(e))

    conversation_simulator = ConversationSimulatorSharded(sample=sample, assistant_model=args.assistant_model, user_model=args.user_model, dataset_fn=dataset_fn, user_agent=user_agent)
    conversation_simulator.run(verbose=True, save_log=True)

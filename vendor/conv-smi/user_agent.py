import json

from agent_config import agent_config
from model_openai import generate_json
from utils import extract_conversation

class UserAgent:
    def __init__(self, task, model="gpt-4o"):
        self.model = model
        self.task = task
        self.task_name = task.get_task_name()
        self.params = agent_config.user()
        with open("prompts/user_agent.txt", "r") as f:
            self.prompt_response = f.read()

    def generate_response(self, conversation, sample, temperature=None):
        num_user_msgs = sum(1 for msg in conversation if msg["role"] == "user")

        if self.task_name in ["translation", "summary", "data2text"]:
            return self.task.populate_sharded_prompt(sample, num_user_msgs)

        if num_user_msgs == 0:
            shard_id = -1
            first_shard = sample["shards"][0]
            shard_id = first_shard["shard_id"]
            initial_query = first_shard["shard"]

            return initial_query, shard_id

        else: # Turn 2 ... n
            # Both reveal log types: the sharded simulator writes
            # `shard_revealed`, the snowball simulator writes `hint_revealed`.
            # Missing either would show the agent an empty revealed list and
            # invite it to reveal the same shard again.
            shard_revealed_ids = [msg["content"]["shard_id"] for msg in conversation if msg["role"] == "log" and msg["content"]["type"] in ("shard_revealed", "hint_revealed")]

            shards = sample["shards"][1:] # filter out the first shard, which is the initial query
            shard_ids = [shard["shard_id"] for shard in shards]
            shard_ids_revealed = [shard_id for shard_id in shard_ids if shard_id in shard_revealed_ids]
            shard_ids_not_revealed = [shard_id for shard_id in shard_ids if shard_id not in shard_revealed_ids]

            # Project shards down to id, text, and attachment *labels*: the
            # `images` entries carry file paths (".../layer_2.png") that would
            # leak layer identity into the simulated user's prompt, but the
            # neutral labels ("layer 2") are already spoken aloud by the
            # "(sharing layer 2)" turn captions, and without them the agent
            # cannot make an informed choice between image-carrying shards.
            def shard_view(shard):
                view = {"shard_id": shard["shard_id"], "shard": shard["shard"]}
                images = shard.get("images") or []
                if images:
                    view["attaches"] = [
                        (image.get("label") or "an image") if isinstance(image, dict) else "an image"
                        for image in images
                    ]
                return view

            shard_texts_revealed = [shard_view(shard) for shard in shards if shard["shard_id"] in shard_ids_revealed]
            shard_texts_not_revealed = [shard_view(shard) for shard in shards if shard["shard_id"] in shard_ids_not_revealed]
            shards_revealed_str = json.dumps(shard_texts_revealed)
            shards_not_revealed_str = json.dumps(shard_texts_not_revealed)


            # hide_image_paths: the trace's image parts carry local file paths
            # whose names (".../layer_2.png") would leak layer identity, the
            # same leak `shard_view` closes for the shard lists.
            conversation_so_far = extract_conversation(conversation, to_str=True, skip_system=True, hide_image_paths=True)
            user_agent_prompt_populated = self.prompt_response.replace("[[CONVERSATION_SO_FAR]]", conversation_so_far).replace("[[SHARDS_REVEALED]]", shards_revealed_str).replace("[[SHARDS_NOT_REVEALED]]", shards_not_revealed_str)
            response_obj = generate_json([{"role": "user", "content": user_agent_prompt_populated}], model=self.model, return_metadata=True, **self.params.call_kwargs(self.model, temperature=temperature))
            response = response_obj["message"]

            # The shard id comes straight out of model JSON, and the
            # simulators' loop guards trust it: an id that is a string, not a
            # real shard, or already revealed would corrupt the revealed-set
            # accounting (at worst looping the conversation forever). Map
            # anything but a fresh, valid id to -1 (reveals nothing).
            response_text = response.get("response", "")
            if not isinstance(response_text, str):
                response_text = json.dumps(response_text)
            try:
                shard_id = int(response.get("shard_id", -1))
            except (TypeError, ValueError):
                shard_id = -1
            valid_ids = {shard["shard_id"] for shard in sample["shards"]}
            if shard_id not in valid_ids or shard_id in shard_revealed_ids:
                shard_id = -1

            return response_text, shard_id

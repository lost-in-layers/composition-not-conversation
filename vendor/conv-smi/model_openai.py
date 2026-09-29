import copy, time, json, re

from model_registry import registry
from multimodal import content_to_text, materialize_local_images, multimodal_content


def _replace_in_content(content, key_string, value):
    if isinstance(content, str):
        return content.replace(key_string, value), key_string in content
    if isinstance(content, list):
        found = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                if key_string in part["text"]:
                    found = True
                    part["text"] = part["text"].replace(key_string, value)
        return content, found
    raise TypeError("message content must be a string or a list of content parts")

def format_messages(messages, variables=None):
    messages = copy.deepcopy(messages)
    variables = variables or {}
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

    for k, v in variables.items():
        key_string = f"[[{k}]]"
        assert type(v) == str, f"[prompt] Variable {k} is not a string"
        last_user_msg["content"], found = _replace_in_content(last_user_msg["content"], key_string, v)
        if not found:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")

    # find all the keys that are still in the prompt using regex [[STR]] where STR is alnum witout space
    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", content_to_text(last_user_msg["content"]))
    if len(keys_still_in_prompt) > 0:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

    return messages

def _parse_json_reply(text):
    """Parse a reply that should be one JSON object; None when it isn't.

    The salvage pass reads only from the *first* `{` — an object fished out
    from deeper inside a malformed reply could be a fragment the model was
    quoting, and grading on that would silently succeed with the wrong
    payload. `raw_decode` ignores trailing text, which covers the common
    ```json fences and prose wrappers.
    """
    # A provider can return `content: null` — a reasoning model whose whole
    # budget went to reasoning, a refusal, a cut-off completion. That is a
    # failed call like any other malformed reply, so it belongs on the retry
    # path. Before this guard `json.loads(None)` raised TypeError, which is not
    # JSONDecodeError, so it flew past every retry and every handler up to the
    # simulator and killed the conversation.
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    start = text.find("{")
    if start == -1:
        return None
    try:
        return json.JSONDecoder().raw_decode(text, start)[0]
    except json.JSONDecodeError:
        return None


def _retry_after_seconds(exc, cap: int = 300):
    """Seconds the server asked us to wait, or None if it did not ask.

    Read from the Retry-After header, and from OpenRouter's error body, which
    repeats it under `error.metadata.headers`. Its presence is what separates a
    temporary throttle from a permanent 4xx: a 402 for an exhausted in-flight
    budget carries one, a 402 for an empty account does not.
    """
    for source in (
        getattr(getattr(exc, "response", None), "headers", None),
        (getattr(exc, "body", None) or {}).get("error", {}).get("metadata", {}).get("headers")
        if isinstance(getattr(exc, "body", None), dict) else None,
    ):
        if not source:
            continue
        try:
            value = source.get("Retry-After") or source.get("retry-after")
        except AttributeError:
            continue
        if value is None:
            continue
        try:
            # Capped: a server asking for an hour should surface as a failure
            # the operator sees, not a worker silently asleep inside a run.
            return max(1, min(int(float(value)), cap))
        except (TypeError, ValueError):
            continue
    return None


class OpenAI_Model:
    def __init__(self):
        self.registry = registry
        # Set this to bypass routing entirely and send every call to one
        # client. Tests use it to stub the transport; nothing else should.
        self.client = None

    def client_for(self, route):
        return self.client if self.client is not None else self.registry.client_for(route.backend)

    # timeout default: multi-image vision prompts and long reasoning
    # completions routinely run past 30s; a timeout that fires on healthy
    # calls just triples the bill via retries.
    def generate(self, messages, model="gpt-4o-mini", timeout=120, max_retries=3, temperature=1.0, top_p=None, top_k=None, is_json=False, return_metadata=False, max_tokens=None, variables=None, images=None, image_base_dir=None):
        route = self.registry.resolve(model)
        kwargs = {}
        # Both nucleus knobs are sent only when configured: an explicit null
        # in agents.yaml has to mean the provider's default, not a literal
        # `top_p: null` in the body, which some servers reject outright.
        if top_p is not None:
            kwargs["top_p"] = top_p
        # `top_k` is not in the OpenAI chat-completions schema, so the client
        # would drop it as an unknown argument; extra_body puts it top-level
        # in the request body, where vLLM and OpenRouter both read it.
        if top_k is not None:
            kwargs["extra_body"] = {"top_k": top_k}
        if is_json:
            kwargs["response_format"] = { "type": "json_object" }
        extra_headers = self.registry.extra_headers(route.backend)
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        N = 0

        messages = copy.deepcopy(messages)
        if images:
            last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]
            last_user_msg["content"] = multimodal_content(last_user_msg["content"], images)
        messages = format_messages(messages, variables)

        # o1- models do not support system message. If the first message is a system message, and the second message is a user message, then prepend the user message with the system message.
        if route.model_id.split("/")[-1].startswith("o1") and len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
            system_message = content_to_text(messages[0]["content"])
            user_content = messages[1]["content"]
            if isinstance(user_content, str):
                messages[1]["content"] = f"System Message: {system_message}\n{user_content}"
            else:
                messages[1]["content"] = copy.deepcopy(user_content)
                messages[1]["content"].insert(0, {"type": "text", "text": f"System Message: {system_message}"})
            messages = messages[1:]

        messages = materialize_local_images(messages, image_base_dir=image_base_dir)

        client = self.client_for(route)
        while True:
            try:
                # `max_tokens` rather than `max_completion_tokens`: both vLLM
                # and OpenRouter accept it, and OpenRouter translates it for
                # the providers that want the newer name.
                response = client.chat.completions.create(model=route.model_id, messages=messages, timeout=timeout, max_tokens=max_tokens, temperature=temperature, **kwargs)
                break
            except Exception as exc:
                N += 1
                # 4xx (other than timeout/rate-limit) means the request
                # itself is wrong; resending it verbatim cannot succeed.
                status = getattr(exc, "status_code", None)
                # ...except when the server says otherwise. OpenRouter answers
                # 402 both for "you are out of credits", which is permanent,
                # and for "this would exceed your in-flight budget, retry in
                # N seconds", which is not — the two are told apart by the
                # Retry-After header the second one carries. Without this, one
                # concurrent-spend throttle killed every conversation in flight
                # on its first attempt, on a run that was otherwise healthy.
                retry_after = _retry_after_seconds(exc)
                throttled = retry_after is not None
                permanent = (
                    isinstance(status, int)
                    and 400 <= status < 500
                    and status not in (408, 429)
                    and not throttled
                )
                if permanent or N >= max_retries:
                    raise RuntimeError(
                        f"Failed to get a chat completion from {route} after {N} attempt(s)"
                    ) from exc
                else:
                    # Honour the server's own number when it gave one; a fixed
                    # 4s backoff against a 120s throttle just burns the retry
                    # budget and fails anyway.
                    time.sleep(retry_after if throttled else 4)

        response = response.to_dict()
        usage = response['usage']
        choice = response["choices"][0]
        response_text = choice["message"].get("content")
        finish_reason = choice.get("finish_reason")
        if response_text is None:
            # Normalised to "" so every caller downstream can assume a string.
            # `finish_reason` rides along because it is what distinguishes the
            # two causes: "length" means the token cap was spent before any
            # content was emitted (a reasoning model on the plain budget), and
            # anything else means the provider returned an empty completion.
            response_text = ""

        if not return_metadata:
            return response_text
        return {"message": response_text, "finish_reason": finish_reason, "total_tokens": usage['total_tokens'], "prompt_tokens": usage['prompt_tokens'], "completion_tokens": usage['completion_tokens']}

    def generate_json(self, messages, model="gpt-4o-mini", **kwargs):
        # Parsing needs the dict shape, so metadata is always requested from
        # `generate`; a caller passing return_metadata=False still gets the
        # parsed payload rather than a crash on indexing a plain string.
        return_metadata = kwargs.pop("return_metadata", True)
        # JSON mode does not guarantee a parseable reply: a model can still
        # fence the object, wrap it in prose, leave an unescaped quote inside
        # a long string, or run past its token cap mid-object. A malformed
        # reply is a failed call exactly like a transport error, so it gets
        # the same retry budget — each attempt is a fresh sample.
        attempts = max(1, kwargs.get("max_retries", 3))
        for _ in range(attempts):
            response = self.generate(messages, model, is_json=True, return_metadata=True, **kwargs)
            parsed = _parse_json_reply(response["message"])
            if parsed is not None:
                response["message"] = parsed
                return response if return_metadata else response["message"]
        reason = response.get("finish_reason")
        if not response["message"]:
            raise RuntimeError(
                f"Model returned an empty completion after {attempts} attempt(s) "
                f"(finish_reason={reason!r}). finish_reason='length' means the "
                f"token cap was spent before any content: add this model to "
                f"`reasoning_models` in agents.yaml so it gets "
                f"reasoning_max_tokens instead of max_tokens."
            )
        raise RuntimeError(
            f"Model returned unparseable JSON after {attempts} attempt(s) "
            f"(finish_reason={reason!r}): {response['message'][:300]!r}"
        )


model = OpenAI_Model()
generate = model.generate
generate_json = model.generate_json





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send one call through the configured routing.")
    parser.add_argument("--model", type=str, default="local-vlm")
    args = parser.parse_args()

    messages = [{"role": "user", "content": "Tell me a joke about UC Berkeley."}]
    print(generate(messages, model=args.model, return_metadata=True))

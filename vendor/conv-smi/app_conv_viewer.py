import json
import os
import streamlit as st
from collections import defaultdict
from datetime import datetime

from tasks import get_task
from multimodal import image_base_dir, image_display_source


def display_message(role, content, timestamp, image_directory=None):
    container = st.chat_message(role)
    if isinstance(content, str):
        container.write(f"{content}\n\n*{timestamp}*")
        return

    for part in content if isinstance(content, list) else []:
        if part.get("type") == "text":
            container.write(part.get("text", ""))
        elif part.get("type") == "image_url":
            container.image(image_display_source(part, image_directory))
    container.write(f"*{timestamp}*")


def layer_verdict(evaluation_return):
    """The grounding half of an Experiment A verdict, if the run had one.

    Answer-only conditions carry no `layer_selection`, so this returns "" and
    their verdict line is unchanged.
    """
    selection = evaluation_return.get("layer_selection")
    if not selection:
        return ""

    if not selection.get("format_ok"):
        return f"\n\nLayers: ⚠️ unreadable ({selection.get('parse_error', 'no layers given')})"

    metrics = evaluation_return.get("layer_metrics") or {}
    predicted = ", ".join(str(i) for i in selection.get("layer_ids") or []) or "none"
    mark = "✅" if metrics.get("exact_match") else "❌"
    parts = [
        f"Layers: {mark} predicted {predicted}",
        f"P/R/F1 {metrics.get('precision', 0):.2f}/{metrics.get('recall', 0):.2f}/{metrics.get('f1', 0):.2f}",
    ]
    if selection.get("confidence") is not None:
        parts.append(f"confidence {selection['confidence']:.2f}")
    if evaluation_return.get("joint_correct"):
        parts.append("jointly correct")
    return "\n\n" + " — ".join(parts)


def load_conversations(file_path):
    conversations = []
    with open(file_path, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))
    return conversations


def format_timestamp(timestamp_str):
    if not timestamp_str:  # Handle empty string
        return "No timestamp"
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%H:%M:%S")
    except ValueError:  # Handle invalid timestamp format
        return "Invalid timestamp"


def group_conversations_by_model(conversations):
    grouped = {}
    for conv in conversations:
        if conv['assistant_model'] not in grouped:
            grouped[conv['assistant_model']] = []
        grouped[conv['assistant_model']].append(conv)
    return grouped


def get_conversation_stats(conversations):
    total_convs = len(conversations)
    if total_convs == 0:
        return 0, 0, 0, 0

    solved_convs = sum(1 for conv in conversations if conv.get('is_correct', False))
    avg_turns = sum(len([t for t in conv["trace"] if t["role"] == "user"]) for conv in conversations) / total_convs
    success_rate = (solved_convs/total_convs)*100 if total_convs > 0 else 0

    return total_convs, solved_convs, success_rate, avg_turns


def display_chat(conversation, sample, image_directory=None):
    st.title(f"Conversation Simulator Viewer ({conversation['conv_id']})")

    # Display conversation metadata
    st.sidebar.header("Conversation Info")
    st.sidebar.write(f"Task ID: {conversation.get('task_id', 'N/A')}")
    st.sidebar.write(f"Assistant Model: {conversation.get('assistant_model', 'N/A')}")
    st.sidebar.write(f"Judge Model: {conversation.get('judge_model', 'N/A')}")
    st.sidebar.write(f"User Model: {conversation.get('user_model', 'N/A')}")
    st.sidebar.write(f"Correct: {conversation.get('is_correct', 'N/A')}")
    num_user_turns = len([t for t in conversation.get('trace', []) if t.get('role') == 'user'])
    st.sidebar.write(f"Number of User Turns: {num_user_turns}")

    # Display the chat messages
    for turn in conversation.get('trace', []):
        role = turn.get('role', '')
        timestamp = format_timestamp(turn.get('timestamp', ''))

        if role == 'user':
            message = turn.get('content', '')
            display_message('user', message, timestamp, image_directory)

        elif role == 'assistant':
            message = turn.get('content', '')
            display_message('assistant', message, timestamp, image_directory)

        elif role == 'log':
            # Optionally display system logs differently
            content = turn.get('content', {})
            log_type = content.get('type', '')
            if log_type == 'answer-evaluation':
                exact_answer = f"```\n{content.get('exact_answer', '')}\n```"
                evaluation_return = content.get('evaluation_return', {}) or {}
                reasoning = evaluation_return.get('reasoning', '')
                grader = evaluation_return.get('grader', '')
                footer = f"\n\n*{reasoning}* ({grader})" if reasoning or grader else ""
                footer += layer_verdict(evaluation_return)
                if "is_correct" in content and (type(content["is_correct"]) == bool or type(content["is_correct"]) == int):
                    is_correct = content.get('is_correct', False)
                    st.chat_message('system').write(f"{exact_answer}\n\nAnswer evaluation: {'✅ Correct' if is_correct else '❌ Incorrect'}{footer}\n\n*{timestamp}*")
                elif "score" in content and type(content["score"]) == float:
                    score = content.get('score', 0)
                    st.chat_message('system').write(f"{exact_answer}\n\nAnswer evaluation: {score}{footer}\n\n*{timestamp}*")

    reference_answer = sample.get("answer", sample.get("Answer"))
    if reference_answer is not None:
        st.chat_message('system').write(f"Reference Answer: {reference_answer}")


def main():
    st.set_page_config(page_title="Conversation Viewer", layout="wide")

    LOG_PATH = "logs"
    st.sidebar.write("="*20)
    st.sidebar.write(f"Showing conversations from **{LOG_PATH}/**")
    st.sidebar.write("="*20)

    try:
        # Get initial options
        tasks = os.listdir(LOG_PATH)

        # Change selectbox to radio for task
        selected_task = st.sidebar.radio(
            "Select Task",
            options=sorted(tasks)
        )


        task = get_task(selected_task)
        dataset_fn = task.get_dataset_file().split("/")[-1]

        # Get available tasks for selected type
        available_conv_types = [d for d in os.listdir(f'{LOG_PATH}/{selected_task}')
                                if os.path.isdir(os.path.join(LOG_PATH, selected_task, d))]

        # Change selectbox to radio for conversation type
        selected_type = st.sidebar.radio("Select Conversation Type", options=sorted(available_conv_types))

        # Get all JSONL files for the selected type and task
        log_path = os.path.join(LOG_PATH, selected_task, selected_type)
        log_files = [f for f in os.listdir(log_path) if f.endswith('.jsonl')]

        # Load all conversations from the selected path
        conversations = []
        for file in log_files:
            file_path = os.path.join(log_path, file)
            for conversation in load_conversations(file_path):
                if conversation["dataset_fn"] == dataset_fn:
                    conversations.append(conversation)

        # Group conversations by assistant model
        grouped_conversations = group_conversations_by_model(conversations)

        # Third selectbox for model
        selected_model = st.sidebar.selectbox(
            "Select Assistant Model",
            options=sorted(grouped_conversations.keys())
        )

        # Get conversations for selected model
        all_model_conversations = grouped_conversations[selected_model]

        # Add a checkbox to filter only the conversations that were not correct
        filter_incorrect = st.sidebar.checkbox("Show only incorrect conversations", value=False)

        if filter_incorrect:
            all_model_conversations = [conv for conv in all_model_conversations if not conv.get("is_correct", True)]

        # Group by task_id
        grouped_by_task_id = defaultdict(list)
        for conv in all_model_conversations:
            grouped_by_task_id[conv["task_id"]].append(conv)
        selected_task_id = st.sidebar.selectbox(
            "Select Task ID",
            options=sorted(grouped_by_task_id.keys())
        )
        model_conversations = grouped_by_task_id[selected_task_id]

        # Rest of the conversation selection and display logic
        conversation_ids = [conv["conv_id"] for conv in model_conversations]

        selected_conv_id = st.sidebar.selectbox(
            "Select Conversation ID",
            options=sorted(conversation_ids)
        )

        # Display statistics for selected model
        st.sidebar.header("Model Statistics")
        total_convs, solved_convs, success_rate, avg_turns = get_conversation_stats(all_model_conversations)

        st.sidebar.write(f"Total Conversations: {total_convs}")
        st.sidebar.write(f"Solved Conversations: {solved_convs}")
        st.sidebar.write(f"Success Rate: {success_rate:.1f}%")
        st.sidebar.write(f"Average Turns: {avg_turns:.1f}")


        selected_conversation = [conv for conv in model_conversations if conv["conv_id"] == selected_conv_id][0]

        sample = task.get_sample(selected_conversation["task_id"])

        display_chat(
            selected_conversation,
            sample,
            image_directory=image_base_dir(task.get_dataset_file()),
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)  # full traceback; a bare message hides the real bug

if __name__ == "__main__":
    main()

import numpy as np

# def apply_message_template(messages, **kwargs):
#     return messages

def get_system_prompt():
    SYSTEM_PROMPT = """You are an expert AI assistant that answers questions about a video by iteratively analyzing it. Provide your detailed reasoning between the <think> </think> tags, and then give your final answer (OPTION only) between the <answer> </answer> tags."""
    return SYSTEM_PROMPT


# def get_image_placeholders(num_frames: int) -> str:
#     return f"<image>" * num_frames

def get_image_placeholders(num_frames: int, total_frames: int) -> str:
    return f"<image>" * num_frames


def apply_message_template(messages, **kwargs):
    assert (
        messages[0]["role"] != "system"
    ), "System message should not be applied to the message template"

    assert len(messages) == 1, "Only one message is allowed"

    total_frames = kwargs["extra_info"]["total_frames"]
    num_frames = len(kwargs["multi_modal_data"]["image"])
    # image_placeholders = get_image_placeholders(num_frames=num_frames)
    image_placeholders = get_image_placeholders(num_frames=num_frames, total_frames=total_frames)

    question = messages[0]["content"]
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": image_placeholders + "\n" + question},
    ]
    return messages
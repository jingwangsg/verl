"""Minimal router that always selects the tool agent loop."""

from typing import Any


def always_tool_agent(sample: Any) -> str:  # noqa: ANN401
    """Route every sample to the "tool_agent" loop."""

    return "tool_agent"

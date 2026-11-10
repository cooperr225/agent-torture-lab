"""The tool registry: one place that lists every tool the agent can use.

Each entry has a description and input schema (what the model sees) and the
Python function to run. To add a tool, add one entry here. Defences can later
wrap or filter this dictionary.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tools import files


@dataclass(frozen=True)
class ToolSpec:
    description: str
    input_schema: dict[str, Any]
    func: Callable[..., str]


def _schema(**properties: str) -> dict[str, Any]:
    """A simple 'object with these required text fields' JSON schema."""
    return {
        "type": "object",
        "properties": {k: {"type": "string", "description": v} for k, v in properties.items()},
        "required": list(properties),
        "additionalProperties": False,
    }


DEFAULT_REGISTRY: dict[str, ToolSpec] = {
    "read_email": ToolSpec(
        "Read one email from the inbox by file name, for example '001.txt'.",
        _schema(name="File name of the email, e.g. 001.txt"),
        files.read_email,
    ),
    "search_web": ToolSpec(
        "Search the web for pages matching some keywords. Returns the text of the best matches.",
        _schema(query="Keywords to search for"),
        files.search_web,
    ),
    "read_note": ToolSpec(
        "Read one of the user's notes by file name, for example 'shopping-list.txt'.",
        _schema(name="File name of the note"),
        files.read_note,
    ),
    "write_note": ToolSpec(
        "Create or overwrite one of the user's notes.",
        _schema(name="File name of the note", content="Full text of the note"),
        files.write_note,
    ),
    "send_message": ToolSpec(
        "Send a message to someone.",
        _schema(to="Who to send the message to", body="The message text"),
        files.send_message,
    ),
}


def tool_definitions(registry: dict[str, ToolSpec]) -> list[dict[str, Any]]:
    """The tool list in the shape the Anthropic API expects."""
    return [
        {"name": name, "description": spec.description, "input_schema": spec.input_schema}
        for name, spec in registry.items()
    ]


def run_tool(registry: dict[str, ToolSpec], name: str, tool_input: Any) -> tuple[str, bool]:
    """Run one tool call. Returns (result_text, is_error). Never raises."""
    spec = registry.get(name)
    if spec is None:
        return f"Error: unknown tool '{name}'.", True
    if not isinstance(tool_input, dict):
        return "Error: tool input must be an object.", True
    # Only pass arguments the schema declares. Tool functions have extra test-only
    # arguments (like 'world') that the model must never be able to set.
    allowed = set(spec.input_schema.get("properties", {}))
    if not set(tool_input) <= allowed:
        return f"Error: wrong arguments for tool '{name}'.", True
    try:
        result = spec.func(**tool_input)
    except TypeError:
        return f"Error: wrong arguments for tool '{name}'.", True
    except Exception:  # a tool bug must not crash the whole run
        return f"Error: tool '{name}' failed.", True
    return result, result.startswith("Error:")

"""The agent loop, built directly on the Anthropic SDK (no framework).

How it works:
1. Send the task and the tool list to the model.
2. If the model asks for tools, run them and send the results back.
3. Repeat until the model gives a final answer or we hit the step limit.

Every run is logged to SQLite (see harness/logging_db.py).
"""

from dataclasses import dataclass

import anthropic

from agent.config import Settings
from agent.pricing import request_cost
from agent.registry import DEFAULT_REGISTRY, ToolSpec, run_tool, tool_definitions
from harness.logging_db import RunLog

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful personal assistant. You can read the user's emails and notes, "
    "search web pages, write notes and send messages using the tools provided. "
    "Use tools when they help, then give the user a short, clear final answer."
)
MAX_TOKENS_PER_REQUEST = 16_000


@dataclass
class RunResult:
    run_id: int
    status: str  # completed, step_limit, refused, truncated, error
    final_answer: str | None
    steps: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


def make_client(settings: Settings) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())


def _text_of(content) -> str:
    return "\n".join(b.text for b in content if b.type == "text").strip()


def run_agent(
    task: str,
    settings: Settings,
    *,
    client=None,
    registry: dict[str, ToolSpec] | None = None,
    log: RunLog | None = None,
    model: str | None = None,
    system: str = DEFAULT_SYSTEM_PROMPT,
) -> RunResult:
    """Run the agent on one task and return the result. Logs everything to SQLite."""
    client = client or make_client(settings)
    registry = DEFAULT_REGISTRY if registry is None else registry
    model = model or settings.agent_model
    own_log = log is None
    log = log or RunLog(settings.db_path)

    run_id = log.start_run(model, task)
    messages: list = [{"role": "user", "content": task}]
    tools = tool_definitions(registry)

    status, final_answer, steps = "step_limit", None, 0
    input_tokens = output_tokens = 0
    cost: float | None = 0.0

    try:
        for step in range(1, settings.max_steps + 1):
            steps = step
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS_PER_REQUEST,
                system=system,
                tools=tools,
                messages=messages,
            )

            in_tok, out_tok = response.usage.input_tokens, response.usage.output_tokens
            input_tokens += in_tok
            output_tokens += out_tok
            step_cost = request_cost(model, in_tok, out_tok)
            cost = None if (cost is None or step_cost is None) else cost + step_cost

            if response.stop_reason == "tool_use":
                # Keep the model's full reply (including any thinking blocks) unchanged.
                messages.append({"role": "assistant", "content": response.content})
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result, is_error = run_tool(registry, block.name, block.input)
                    log.log_tool_call(run_id, step, block.name, block.input, result, is_error)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                        **({"is_error": True} if is_error else {}),
                    })
                # All results go back together in ONE user message.
                messages.append({"role": "user", "content": results})
                continue

            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue

            final_answer = _text_of(response.content) or None
            status = {"end_turn": "completed", "refusal": "refused", "max_tokens": "truncated"}.get(
                response.stop_reason, "completed"
            )
            break
    except Exception as exc:
        # Record the failure (type only: never the message, which could hold sensitive text).
        status, final_answer = "error", f"{type(exc).__name__}"
        log.finish_run(run_id, status, final_answer, steps, input_tokens, output_tokens, cost)
        if own_log:
            log.close()
        raise

    log.finish_run(run_id, status, final_answer, steps, input_tokens, output_tokens, cost)
    if own_log:
        log.close()
    return RunResult(run_id, status, final_answer, steps, input_tokens, output_tokens, cost)

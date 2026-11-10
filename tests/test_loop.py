"""Tests for the agent loop, registry, pricing and run logging.

No real API calls: a fake client plays back scripted model replies.
"""

import json
from types import SimpleNamespace

import pytest

from agent.config import Settings
from agent.loop import run_agent
from agent.pricing import request_cost
from agent.registry import DEFAULT_REGISTRY, ToolSpec, run_tool, tool_definitions
from harness.logging_db import RunLog

# --- fake model ------------------------------------------------------------

def usage(i=100, o=50):
    return SimpleNamespace(input_tokens=i, output_tokens=o)


def text_reply(text, stop="end_turn", i=100, o=50):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason=stop, usage=usage(i, o)
    )


def tool_reply(*calls, i=100, o=50):
    """calls: (id, tool name, input dict) tuples."""
    blocks = [SimpleNamespace(type="tool_use", id=c[0], name=c[1], input=c[2]) for c in calls]
    return SimpleNamespace(content=blocks, stop_reason="tool_use", usage=usage(i, o))


class FakeClient:
    """Plays back scripted replies and records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        kwargs["messages"] = list(kwargs["messages"])  # snapshot: the loop keeps appending
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def settings(tmp_path):
    return Settings(anthropic_api_key="sk-test-key-123", max_steps=10, db_path=tmp_path / "runs.db")


@pytest.fixture
def log(settings):
    log = RunLog(settings.db_path)
    yield log
    log.close()


def echo_registry():
    return {
        "echo": ToolSpec(
            "Echo text back.",
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            lambda text: f"echo: {text}",
        )
    }


# --- the loop --------------------------------------------------------------

def test_simple_run_without_tools(settings, log):
    client = FakeClient(text_reply("Hello!", i=1000, o=500))
    result = run_agent("Say hi", settings, client=client, log=log)

    assert (result.status, result.final_answer, result.steps) == ("completed", "Hello!", 1)
    assert (result.input_tokens, result.output_tokens) == (1000, 500)
    assert result.cost_usd == pytest.approx(0.00035)  # haiku: 1000*0.10/1M + 500*0.50/1M

    row = log.get_run(result.run_id)
    assert (row["task"], row["model"], row["status"]) == ("Say hi", "claude-haiku-5-5", "completed")
    assert row["final_answer"] == "Hello!" and row["cost_usd"] == pytest.approx(0.00035)


def test_tool_call_is_run_logged_and_result_sent_back(settings, log):
    client = FakeClient(
        tool_reply(("t1", "echo", {"text": "hi"})),
        text_reply("Done."),
    )
    result = run_agent("Use echo", settings, client=client, registry=echo_registry(), log=log)

    assert result.status == "completed" and result.steps == 2
    assert (result.input_tokens, result.output_tokens) == (200, 100)  # summed over both requests

    [call] = log.get_tool_calls(result.run_id)
    assert (call["step"], call["tool_name"], call["result"], call["is_error"]) == (1, "echo", "echo: hi", 0)
    assert json.loads(call["arguments"]) == {"text": "hi"}

    # The second request carried the tool result, tied to the right tool_use id.
    last = client.calls[1]["messages"][-1]
    assert last["role"] == "user"
    assert last["content"] == [{"type": "tool_result", "tool_use_id": "t1", "content": "echo: hi"}]


def test_several_tool_calls_in_one_turn_return_in_one_message(settings, log):
    client = FakeClient(
        tool_reply(("a", "echo", {"text": "1"}), ("b", "echo", {"text": "2"})),
        text_reply("ok"),
    )
    result = run_agent("two", settings, client=client, registry=echo_registry(), log=log)
    results = client.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert len(log.get_tool_calls(result.run_id)) == 2


def test_step_limit_stops_a_runaway_agent(settings, log):
    settings.max_steps = 3
    client = FakeClient(*[tool_reply((f"t{i}", "echo", {"text": "again"})) for i in range(10)])
    result = run_agent("loop forever", settings, client=client, registry=echo_registry(), log=log)

    assert (result.status, result.steps, result.final_answer) == ("step_limit", 3, None)
    assert len(client.calls) == 3
    assert log.get_run(result.run_id)["status"] == "step_limit"
    assert len(log.get_tool_calls(result.run_id)) == 3


def test_default_step_limit_is_ten(settings, log):
    client = FakeClient(*[tool_reply((f"t{i}", "echo", {"text": "x"})) for i in range(30)])
    result = run_agent("loop", settings, client=client, registry=echo_registry(), log=log)
    assert result.steps == 10 and len(client.calls) == 10


def test_tool_errors_go_back_to_the_model_instead_of_crashing(settings, log):
    client = FakeClient(
        tool_reply(("t1", "nope", {}), ("t2", "echo", {"wrong": "args"})),
        text_reply("I could not do that."),
    )
    result = run_agent("bad tools", settings, client=client, registry=echo_registry(), log=log)

    assert result.status == "completed"
    sent = client.calls[1]["messages"][-1]["content"]
    assert all(r["is_error"] is True for r in sent)
    assert "unknown tool" in sent[0]["content"] and "wrong arguments" in sent[1]["content"]
    assert [c["is_error"] for c in log.get_tool_calls(result.run_id)] == [1, 1]


def test_refusal_and_truncation_are_recorded(settings, log):
    r1 = run_agent("a", settings, client=FakeClient(text_reply("", stop="refusal")), log=log)
    r2 = run_agent("b", settings, client=FakeClient(text_reply("cut", stop="max_tokens")), log=log)
    assert (r1.status, r2.status) == ("refused", "truncated")


def test_api_failure_is_logged_then_raised(settings, log):
    client = FakeClient(tool_reply(("t1", "echo", {"text": "x"})), RuntimeError("boom sk-secret"))
    with pytest.raises(RuntimeError):
        run_agent("fail", settings, client=client, registry=echo_registry(), log=log)

    row = log.get_run(1)
    assert row["status"] == "error" and row["steps"] == 2
    assert row["final_answer"] == "RuntimeError"  # error type only, never the message
    assert len(log.get_tool_calls(1)) == 1  # earlier work is still recorded


def test_run_uses_requested_model_system_prompt_and_tools(settings, log):
    client = FakeClient(text_reply("ok"))
    run_agent("x", settings, client=client, log=log, model="claude-sonnet-5-5", system="SYS")
    call = client.calls[0]
    assert call["model"] == "claude-sonnet-5-5" and call["system"] == "SYS"
    assert {t["name"] for t in call["tools"]} == set(DEFAULT_REGISTRY)
    assert log.get_run(1)["model"] == "claude-sonnet-5-5"


def test_api_key_is_never_written_to_the_database(settings, log):
    run_agent("x", settings, client=FakeClient(text_reply("ok")), log=log)
    dump = "\n".join(log.conn.iterdump())
    assert "sk-test-key-123" not in dump


def test_runs_are_numbered_and_kept_separate(settings, log):
    for task in ("one", "two"):
        run_agent(task, settings, client=FakeClient(text_reply(task)), log=log)
    assert [log.get_run(i)["task"] for i in (1, 2)] == ["one", "two"]


# --- registry --------------------------------------------------------------

def test_registry_lists_the_five_tools_and_hides_the_world_argument():
    defs = {d["name"]: d for d in tool_definitions(DEFAULT_REGISTRY)}
    assert set(defs) == {"read_email", "search_web", "read_note", "write_note", "send_message"}
    for d in defs.values():
        assert "world" not in d["input_schema"]["properties"]
        assert d["input_schema"]["additionalProperties"] is False


def test_run_tool_cannot_be_given_the_world_argument():
    # The model must not be able to repoint a tool at another folder.
    for tool, args in [
        ("read_email", {"name": "001.txt", "world": "/etc"}),
        ("write_note", {"name": "x.txt", "content": "hi", "world": "/tmp"}),
        ("send_message", {"to": "a", "body": "b", "world": "/tmp"}),
    ]:
        result, is_error = run_tool(DEFAULT_REGISTRY, tool, args)
        assert is_error and "wrong arguments" in result


def test_run_tool_runs_real_tools_on_the_sample_data():
    result, is_error = run_tool(DEFAULT_REGISTRY, "read_email", {"name": "001.txt"})
    assert not is_error and "Subject:" in result
    result, is_error = run_tool(DEFAULT_REGISTRY, "read_email", {"name": "../../etc/passwd"})
    assert is_error and result.startswith("Error:")


def test_run_tool_survives_a_crashing_tool():
    def boom(**_):
        raise ValueError("secret detail")

    registry = {"boom": ToolSpec("x", {"type": "object", "properties": {}}, boom)}
    result, is_error = run_tool(registry, "boom", {})
    assert is_error and "secret detail" not in result


def test_run_tool_rejects_non_object_input():
    assert run_tool(echo_registry(), "echo", "just a string")[1] is True


# --- pricing ---------------------------------------------------------------

def test_prices():
    assert request_cost("claude-haiku-5-5", 100_000, 100_000) == pytest.approx(0.06)
    assert request_cost("claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(12.00)


def test_haiku_long_prompt_tier_costs_more():
    normal = request_cost("claude-haiku-5-5", 100_000, 1000)
    long = request_cost("claude-haiku-5-5", 100_001, 1000)
    assert normal == pytest.approx(100_000 * 0.10 / 1e6 + 1000 * 0.50 / 1e6)
    assert long == pytest.approx(100_001 * 0.50 / 1e6 + 1000 * 2.50 / 1e6)


def test_unknown_model_has_no_price_and_cost_is_null(settings, log):
    assert request_cost("some-new-model", 10, 10) is None
    result = run_agent("x", settings, client=FakeClient(text_reply("ok")), log=log, model="some-new-model")
    assert result.cost_usd is None and log.get_run(result.run_id)["cost_usd"] is None

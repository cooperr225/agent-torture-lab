# Agent Torture Lab

## What this project is
An automated red-teaming harness for a small, tool-using LLM agent. We attack a target agent (prompt injection, data exfiltration, tool misuse, goal drift, bad input), measure how often attacks succeed, and test which defences reduce failures and what they cost in normal-task performance.

**Research question:** Which attacks work against a realistic multi-tool agent, which defences actually reduce failures, and what does each defence cost in capability?

## Folder layout
- `agent/` the target agent: a tool-use loop built directly on the Anthropic SDK (no framework)
- `tools/` the agent's tools; every tool works only inside `mock_world/`
- `mock_world/` fake emails, web pages, notes and calendar the agent can read
- `attacks/` attack cases as data: `seed/` (hand-written), `generated/` (model-made variants), `heldout/` (50 cases, locked until the final run)
- `harness/` runner, verdict rules, logging to SQLite
- `defences/` switchable defences: D1 prompt hardening, D2 content marking, D3 permissions layer, D4 second-model check
- `reports/` charts, tables and the write-up
- `devlog/` one markdown entry per working session
- `tests/` pytest tests

## Rules
- All tools operate only inside `mock_world/`. Never touch real accounts, real email or the open internet from the agent.
- Never use real credentials. API keys come from `.env` and are never committed or printed.
- `send_message` writes to a log file. It never sends anything.
- Write a small pytest test for every tool and every verdict rule.
- Prefer rule-based verdicts (canary leaked, disallowed tool call, file changed) over LLM judging.
- Do not read or edit `attacks/heldout/` until I say the final evaluation run has started.
- After each task, explain what you changed in two sentences.
- Ask for a plan before large changes. One task per prompt.
- Models: the target agent defaults to claude-haiku-5-5; the judge and the stronger target use claude-sonnet-5-5.
- Python 3.13, managed with uv. Run things with `uv run`.

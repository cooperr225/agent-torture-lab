# Agent Torture Lab

An automated harness that attacks a small tool-using AI agent (prompt injection, data exfiltration, tool misuse, goal drift, bad input), measures how often each attack works, and tests which defences help and what they cost.

**Status:** setup. Results and report to come.

## Research question

Which attacks work against a realistic multi-tool agent, which defences actually reduce failures, and what does each defence cost in capability?

## Safety

The agent only ever touches fake data in `mock_world/` and runs in a sandbox. Nothing here connects to a real account.

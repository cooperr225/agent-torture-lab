"""Price table and cost calculation.

Prices are US dollars per 1 million tokens, from Anthropic's published rates
(checked 2026-11-10 against the claude-api reference, cached 2026-10-06).
Prices change, so re-check them before the final evaluation run.

Prompt caching is not used by this agent, so only plain input and output
tokens are counted.
"""

# model -> (input $/1M, output $/1M, long_prompt_threshold, long_input, long_output)
# Haiku 5.5 charges more when ONE request's prompt is over 100K tokens.
PRICES: dict[str, tuple[float, float, int | None, float | None, float | None]] = {
    "claude-haiku-5-5": (0.10, 0.50, 100_000, 0.50, 2.50),
    "claude-sonnet-5-5": (2.00, 10.00, None, None, None),
}


def request_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost in dollars of ONE API request, or None if the model has no price listed."""
    if model not in PRICES:
        return None
    in_price, out_price, threshold, long_in, long_out = PRICES[model]
    if threshold is not None and input_tokens > threshold:
        in_price, out_price = long_in, long_out
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000

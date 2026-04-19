# src/insights.py
#
# AI-generated narrative insights for DemandIQ using Google Gemini.
#
# The LLM receives pre-computed metric values and is explicitly instructed to
# narrate and interpret only — it must never recalculate or invent numbers.
# All quantitative facts are supplied by the caller via `metrics_dict`.
#
# The model is asked to return a simple labelled plain-text format (not JSON)
# so that output parsing is reliable even when max_output_tokens is small.
#
# Expected model output format:
#
#   INSIGHT_PARAGRAPH:
#   <3–5 sentence narrative>
#
#   RECOMMENDATIONS:
#   - <recommendation 1>
#   - <recommendation 2>
#   - <recommendation 3>
#
# Public API:
#   build_prompt(metrics_dict)              -> str
#   generate_insights(metrics_dict)         -> dict
#
# Returned dict shape:
#   {
#       "insight_paragraph": "...",
#       "recommendations":  ["...", "...", "..."],
#   }
#
# The module is self-contained and ready to import from a Streamlit page.

import logging

import google.generativeai as genai

from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# ── Model configuration ────────────────────────────────────────────────────────
_MODEL_NAME = "gemini-2.5-flash"

# Keys the caller may include in metrics_dict.  Only keys present are injected
# into the prompt; unknown keys are appended generically.
_KNOWN_KEYS = {
    "top_category",
    "top_state",
    "revenue_trend_direction",
    "forecast_next_30_days",
    "avg_order_value",
    "total_revenue",
}

# Fallback response returned when the API key is absent or a call fails.
_FALLBACK_RESPONSE: dict = {
    "insight_paragraph": (
        "Insights are unavailable at the moment. "
        "Please ensure that GEMINI_API_KEY is set in your .env file and try again."
    ),
    "recommendations": [
        "Set GEMINI_API_KEY in your .env file to enable AI-generated insights.",
        "Verify that the google-generativeai package is installed and up to date.",
        "Re-run the dashboard after fixing the API key configuration.",
    ],
}


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(metrics_dict: dict) -> str:
    """
    Construct the Gemini prompt from the provided metrics dictionary.

    The model is instructed to return plain labelled text (not JSON or
    markdown) so that parsing is robust regardless of output token limits.

    Parameters
    ----------
    metrics_dict : dict
        Pre-computed metrics, e.g.:
        {
            "total_revenue":           13_315_828.19,
            "avg_order_value":         137.53,
            "top_category":            "health_beauty",
            "top_state":               "SP",
            "revenue_trend_direction": "upward",
            "forecast_next_30_days":   823_450.0,
        }
        Any subset of the known keys is accepted; unknown keys are appended.

    Returns
    -------
    str
        The fully-formatted prompt string ready to send to Gemini.
    """
    # ── Format each available metric as a labelled fact line ──────────────────
    fact_lines: list[str] = []

    if "total_revenue" in metrics_dict:
        val = metrics_dict["total_revenue"]
        fact_lines.append(f"- Total historical revenue: R$ {val:,.2f}")

    if "avg_order_value" in metrics_dict:
        val = metrics_dict["avg_order_value"]
        fact_lines.append(f"- Average order value: R$ {val:,.2f}")

    if "top_category" in metrics_dict:
        fact_lines.append(
            f"- Top product category by revenue: {metrics_dict['top_category']}"
        )

    if "top_state" in metrics_dict:
        fact_lines.append(
            f"- Top customer state by revenue: {metrics_dict['top_state']}"
        )

    if "revenue_trend_direction" in metrics_dict:
        fact_lines.append(
            f"- Recent revenue trend direction: {metrics_dict['revenue_trend_direction']}"
        )

    if "forecast_next_30_days" in metrics_dict:
        val = metrics_dict["forecast_next_30_days"]
        # Format numerics as currency; pass strings through unchanged.
        if isinstance(val, (int, float)):
            fact_lines.append(
                f"- Forecasted revenue for the next 30 days: R$ {val:,.2f}"
            )
        else:
            fact_lines.append(
                f"- Forecast outlook for the next 30 days: {val}"
            )

    # Append any extra keys the caller passed that are not in _KNOWN_KEYS.
    for key, value in metrics_dict.items():
        if key not in _KNOWN_KEYS:
            fact_lines.append(f"- {key.replace('_', ' ').title()}: {value}")

    facts_block = "\n".join(fact_lines) if fact_lines else "  (no metrics provided)"

    prompt = f"""You are a senior e-commerce demand analyst writing a concise executive summary for a business intelligence dashboard.

You have been given the following pre-computed metrics about a Brazilian e-commerce business:

{facts_block}

Your task:
1. Write a single cohesive insight paragraph (3–5 sentences) that narrates the business situation. Highlight what is performing well, what patterns are visible, and what the forecast implies for near-term operations.
2. Provide exactly 3 actionable recommendations the business should consider based on these facts.

IMPORTANT RULES:
- Do NOT invent, estimate, or recalculate any numbers beyond what is provided above.
- Refer only to the values listed in the metrics block when citing figures.
- Keep the tone professional, concise, and data-driven.
- Do not use the phrase "I" or "As an AI".
- Do NOT use markdown code fences or return JSON.
- Do NOT include any extra headings or commentary beyond the format below.

Respond using this EXACT format with no deviations:

INSIGHT_PARAGRAPH:
<your 3–5 sentence narrative here>

RECOMMENDATIONS:
- <recommendation 1>
- <recommendation 2>
- <recommendation 3>"""

    return prompt


# ── Structured plain-text parser ───────────────────────────────────────────────

def _parse_structured_response(text: str) -> dict:
    """
    Parse the model's plain-text labelled response into a Python dict.

    Expected input format::

        INSIGHT_PARAGRAPH:
        <3–5 sentence narrative>

        RECOMMENDATIONS:
        - <recommendation 1>
        - <recommendation 2>
        - <recommendation 3>

    Parameters
    ----------
    text : str
        Raw text returned by the model (after stripping leading/trailing
        whitespace).

    Returns
    -------
    dict
        Keys: ``"insight_paragraph"`` (str) and ``"recommendations"``
        (list[str], first 3 items only).

    Raises
    ------
    ValueError
        If either ``INSIGHT_PARAGRAPH:`` or ``RECOMMENDATIONS:`` is absent
        from the text, or if no bullet points can be found.
    """
    text = text.strip()

    # ── Locate the two required section headers ────────────────────────────────
    ip_marker = "INSIGHT_PARAGRAPH:"
    rec_marker = "RECOMMENDATIONS:"

    ip_pos = text.find(ip_marker)
    rec_pos = text.find(rec_marker)

    if ip_pos == -1:
        raise ValueError(
            f"Model response is missing the '{ip_marker}' section header.\n"
            f"Raw output: {text[:300]!r}"
        )
    if rec_pos == -1:
        raise ValueError(
            f"Model response is missing the '{rec_marker}' section header.\n"
            f"Raw output: {text[:300]!r}"
        )
    if ip_pos >= rec_pos:
        raise ValueError(
            f"'{ip_marker}' must appear before '{rec_marker}' in the response.\n"
            f"Raw output: {text[:300]!r}"
        )

    # ── Extract insight paragraph ──────────────────────────────────────────────
    # Everything between the end of INSIGHT_PARAGRAPH: and the start of
    # RECOMMENDATIONS:, with surrounding whitespace stripped.
    insight_block = text[ip_pos + len(ip_marker): rec_pos].strip()
    if not insight_block:
        raise ValueError(
            f"The '{ip_marker}' section is empty.\n"
            f"Raw output: {text[:300]!r}"
        )

    # ── Extract recommendation bullets ────────────────────────────────────────
    rec_block = text[rec_pos + len(rec_marker):].strip()
    bullets: list[str] = []
    for line in rec_block.splitlines():
        line = line.strip()
        if line.startswith("-"):
            # Strip the leading dash and any trailing whitespace.
            bullet_text = line.lstrip("-").strip()
            if bullet_text:
                bullets.append(bullet_text)

    if not bullets:
        raise ValueError(
            f"No bullet points found under '{rec_marker}'.\n"
            f"Raw output: {text[:300]!r}"
        )

    return {
        "insight_paragraph": insight_block,
        # Return only the first 3 recommendations as specified.
        "recommendations": bullets[:3],
    }


# ── Gemini call ────────────────────────────────────────────────────────────────

def generate_insights(metrics_dict: dict) -> dict:
    """
    Generate AI-written narrative insights from pre-computed metrics.

    Builds a structured prompt via :func:`build_prompt`, calls the Gemini
    API, and parses the plain-text response via
    :func:`_parse_structured_response`.  Falls back to a static response on
    any configuration or API error so the dashboard never crashes.

    Parameters
    ----------
    metrics_dict : dict
        Pre-computed metrics. See :func:`build_prompt` for accepted keys.

    Returns
    -------
    dict
        Keys: ``"insight_paragraph"`` (str) and ``"recommendations"``
        (list[str]).  On error, ``_FALLBACK_RESPONSE`` is returned with
        an explanatory message.
    """
    # ── Guard: API key must be configured ─────────────────────────────────────
    if not GEMINI_API_KEY:
        logger.warning(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file to enable AI insights."
        )
        return _FALLBACK_RESPONSE.copy()

    # ── Guard: at least one metric must be present ────────────────────────────
    if not metrics_dict:
        logger.warning("generate_insights() called with an empty metrics_dict.")
        return _FALLBACK_RESPONSE.copy()

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = build_prompt(metrics_dict)
    logger.debug("Gemini prompt built (%d chars).", len(prompt))

    # ── Configure and call Gemini ─────────────────────────────────────────────
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(_MODEL_NAME)

        logger.info("Calling Gemini (%s) for insights …", _MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,        # low temp → consistent, factual tone
                max_output_tokens=1536, # generous headroom for paragraph + 3 bullets
                # No response_mime_type — plain text avoids JSON truncation.
            ),
        )
        raw_text: str = response.text.strip()
        logger.info("Gemini response received (%d chars).", len(raw_text))

    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini API call failed: %s", exc)
        fallback = _FALLBACK_RESPONSE.copy()
        fallback["insight_paragraph"] = (
            f"Insights could not be generated due to an API error: {exc}. "
            "Check your API key and network connection."
        )
        return fallback

    # ── Parse structured plain-text response ──────────────────────────────────
    try:
        result = _parse_structured_response(raw_text)
    except ValueError as exc:
        logger.error("Failed to parse Gemini response: %s", exc)
        logger.debug("Raw Gemini response:\n%s", raw_text)
        fallback = _FALLBACK_RESPONSE.copy()
        fallback["insight_paragraph"] = (
            "The AI returned a response in an unexpected format. "
            "Raw output: " + raw_text[:300]
        )
        return fallback

    logger.info("Insights parsed successfully.")
    return result


# ── Quick smoke-test (run as script) ──────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
        stream=sys.stdout,
    )

    sample_metrics = {
        "total_revenue":           13_315_828.19,
        "avg_order_value":         137.53,
        "top_category":            "health_beauty",
        "top_state":               "SP",
        "revenue_trend_direction": "upward",
        "forecast_next_30_days":   823_450.00,
    }

    print("── Prompt preview ──────────────────────────────────────────────────")
    print(build_prompt(sample_metrics))
    print("\n── Generating insights … ──────────────────────────────────────────")

    result = generate_insights(sample_metrics)

    print("\n── Result ──────────────────────────────────────────────────────────")
    print(json.dumps(result, indent=2, ensure_ascii=False))

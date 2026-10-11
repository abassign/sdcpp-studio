#!/usr/bin/env python3
"""
I2T Response Parser - Shared parser for multimodal model outputs.

Different I2T models wrap their thinking/answer in different formats.
This module provides a single parser that handles all the formats
we currently support, so each I2T worker (PyTorch, llama-server, cpp)
produces the same (thinking, response) split regardless of model family.

Supported formats (handled in priority order):
  1. Gemma 4 / 3 channel markers:
       <|channel|>thought
       ...thinking...
       <|channel|>
       ...answer...
  2. The prefill "empty thinking" pattern that Gemma 4 emits when
     enable_thinking is false in the chat template:
       <|channel|>thought
       <|channel|>
       ...answer...
  3. Legacy app markers:
       ###THINKING###
       ...thinking...
       ###ANSWER###
       ...answer...
  4. JSON: {"thinking": "...", "answer": "..."} (with italian aliases)
  5. Raw text fallback (everything becomes the answer)

Returns:
    (thinking: str, response: str) - both stripped of surrounding
    whitespace. response is NEVER empty - if the model only produced
    thinking, the thinking text is returned as the response (better
    than showing an empty result to the user).
"""

import json
import re

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Gemma 4 / Gemma 3 channel markers
# The opening token is <|channel|>thought (sometimes <|channel>thought, or
# <|channel>analysis depending on the model), and the closing token is
# <|channel|>. We deliberately keep this case-insensitive and tolerant.
_GEMMA_CHANNEL_RE = re.compile(
    r"<\|?channel\|?>\s*thought\s*(.*?)<\|?channel\|?>(.*)$",
    re.IGNORECASE | re.DOTALL,
)

# Same as above but with the "analysis" channel name used by some Gemma
# variants (e.g. google/gemma-3-27b-it with thinking enabled).
_GEMMA_CHANNEL_ALT_RE = re.compile(
    r"<\|?channel\|?>\s*(?:analysis|reasoning)\s*(.*?)<\|?channel\|?>(.*)$",
    re.IGNORECASE | re.DOTALL,
)

# Qwen-style "<think>...</think>" markers
_QWEN_THINK_RE = re.compile(
    r"<think>(.*?)</think>(.*)$",
    re.IGNORECASE | re.DOTALL,
)

# Legacy app markers
_APP_THINK_MARKER = "###THINKING###"
_APP_ANSWER_MARKER = "###ANSWER###"


def parse_model_output(raw_output: str) -> tuple[str, str]:
    """
    Split a multimodal model's raw text output into (thinking, response).

    Args:
        raw_output: The raw `content` string returned by the model
                    (e.g. `result["choices"][0]["message"]["content"]`).

    Returns:
        (thinking, response) - both stripped. response is always non-empty
        when raw_output is non-empty.
    """
    if raw_output is None:
        return "", ""

    text = raw_output.strip()
    if not text:
        return "", ""

    # ------------------------------------------------------------------
    # 1. Gemma 4 / Gemma 3 channel markers
    # ------------------------------------------------------------------
    for pattern in (_GEMMA_CHANNEL_RE, _GEMMA_CHANNEL_ALT_RE):
        m = pattern.search(text)
        if m:
            thinking = m.group(1).strip()
            response = m.group(2).strip()
            # The prefill-empty-thinking case: response is empty, so
            # return the thinking content as the response instead so the
            # user actually sees something.
            if not response and thinking:
                return "", thinking
            return thinking, response

    # ------------------------------------------------------------------
    # 2. Qwen-style <think>...</think>
    # ------------------------------------------------------------------
    m = _QWEN_THINK_RE.search(text)
    if m:
        thinking = m.group(1).strip()
        response = m.group(2).strip()
        if not response and thinking:
            return "", thinking
        return thinking, response

    # ------------------------------------------------------------------
    # 3. Legacy app markers: ###THINKING### ... ###ANSWER### ...
    # ------------------------------------------------------------------
    if _APP_THINK_MARKER in text:
        parts = text.split(_APP_THINK_MARKER, 1)
        if len(parts) == 2:
            tail = parts[1]
            if _APP_ANSWER_MARKER in tail:
                t, r = tail.split(_APP_ANSWER_MARKER, 1)
                return t.strip(), r.strip()
            # Only thinking marker, no answer marker → treat the rest
            # as the answer
            return "", tail.strip()

    # ------------------------------------------------------------------
    # 4. JSON fallback
    # ------------------------------------------------------------------
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            thinking = (
                data.get("thinking")
                or data.get("reasoning")
                or data.get("thought")
                or data.get("ragionamento")
                or ""
            )
            response = (
                data.get("answer")
                or data.get("response")
                or data.get("risposta")
                or data.get("output")
                or ""
            )
            # Normalise escape sequences that some models leave in the
            # JSON-decoded strings.
            thinking = str(thinking).replace("\\n", "\n").strip()
            response = str(response).replace("\\n", "\n").strip()
            if response:
                return thinking, response
    except (json.JSONDecodeError, ValueError):
        pass

    # ------------------------------------------------------------------
    # 5. Raw fallback: no recognised structure, treat everything as
    #    the answer.
    # ------------------------------------------------------------------
    return "", text


def build_chat_template_kwargs(enable_thinking: bool | None) -> dict:
    """
    Build the `chat_template_kwargs` payload to send to llama-server.

    Different model families interpret the chat template differently:

    - Gemma 4 (HauhauCS, google): uses `enable_thinking`. When `false`,
      the chat template prefills `<|channel>thought\\n<channel|>` to make
      the model skip its reasoning phase and start directly with the
      answer.
    - Qwen 3.x: the chat template is harmless to receive this kwarg
      (it is just ignored), so we always pass it.

    Returns:
        Empty dict if `enable_thinking is None` (don't change the model
        behaviour), otherwise `{"enable_thinking": bool(enable_thinking)}`.
    """
    if enable_thinking is None:
        return {}
    return {"enable_thinking": bool(enable_thinking)}

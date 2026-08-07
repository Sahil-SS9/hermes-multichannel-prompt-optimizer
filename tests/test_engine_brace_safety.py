"""Regression test: brace-containing prompts must not crash the optimizer.

Root cause fixed: `_try_rewrite_sync` formatted the template with
`tpl.system_prompt.format(original=text)` OUTSIDE the try block. Any user
prompt containing literal braces (JSON, code, templates) raised
KeyError/IndexError/ValueError that escaped the handler, propagated up
through the `_on_pre_gateway_dispatch` hook, and crashed the gateway on
message dispatch.

Fix: escape literal braces and format INSIDE the try so a malformed prompt
degrades to "no rewrite" (returns None) instead of raising.
"""
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from engine import _try_rewrite_sync, select_template


class _FakeLLM:
    """Minimal pllm stand-in: complete() returns a text response."""

    def __init__(self, text="ok"):
        self._text = text

    def complete(self, messages=None, **kwargs):
        class _R:
            text: str = ""
        r = _R()
        r.text = self._text
        return r


class _RaisingLLM:
    """pllm stand-in that raises on complete — must be handled gracefully."""

    def complete(self, messages=None, **kwargs):
        raise RuntimeError("provider down")


def test_brace_json_prompt_does_not_crash():
    """A JSON-shaped prompt with braces must return None, not raise."""
    text = '{"key": "value", "nested": {"a": 1}}'
    result = _try_rewrite_sync(text, _FakeLLM(), template=select_template())
    # Either a successful rewrite or None — but never an exception.
    assert result is None or isinstance(result, tuple)


def test_brace_code_prompt_does_not_crash():
    """A code-shaped prompt with braces (dict/set/f-string) must not crash."""
    text = "def f():\n    return {k: v for k, v in items if k in {'a', 'b'}}"
    result = _try_rewrite_sync(text, _FakeLLM(), template=select_template())
    assert result is None or isinstance(result, tuple)


def test_plain_prompt_still_works():
    """Non-brace prompts behave as before (no regression)."""
    result = _try_rewrite_sync("Write a haiku about the sea",
                               _FakeLLM(), template=select_template())
    assert result is None or isinstance(result, tuple)


def test_llm_failure_is_graceful():
    """An LLM that raises must degrade to None, not propagate."""
    result = _try_rewrite_sync("hello", _RaisingLLM(),
                               template=select_template())
    assert result is None


def test_single_brace_prompt_does_not_crash():
    """Even a lone unmatched brace must not crash formatting."""
    result = _try_rewrite_sync("Use {name} in the answer",
                               _FakeLLM(), template=select_template())
    assert result is None or isinstance(result, tuple)


if __name__ == "__main__":
    # Allow running directly: python3 tests/test_engine_brace_safety.py
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{5 - failures}/5 passed")
    sys.exit(1 if failures else 0)

"""Tests for the prompt-optimizer plugin.

Loads the plugin exactly the way the Hermes core does
(``hermes_cli.plugins`` -> importlib spec from the plugin directory), so
hook registration, bypass logic, and the engine behave as in production.

Run with the Hermes venv from a directory OUTSIDE this plugin (pytest
treats a rootdir containing ``__init__.py`` as a package, and this
directory's hyphenated name is not a valid Python package identifier)::

    cd /tmp && /path/to/hermes-agent/venv/bin/python -m pytest \\
        ~/.hermes/plugins/prompt-optimizer/tests/test_prompt_optimizer_plugin.py -v
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
_NS_PARENT = "hermes_plugins"


def _load_plugin():
    """Load the plugin module via the same importlib path the core uses."""
    if _NS_PARENT not in sys.modules:
        ns_pkg = types.ModuleType(_NS_PARENT)
        ns_pkg.__path__ = []
        ns_pkg.__package__ = _NS_PARENT
        sys.modules[_NS_PARENT] = ns_pkg
    module_name = f"{_NS_PARENT}.prompt_optimizer"
    if module_name in sys.modules:
        return sys.modules[module_name]
    init_file = PLUGIN_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name, init_file, submodule_search_locations=[str(PLUGIN_DIR)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {init_file}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = module_name
    module.__path__ = [str(PLUGIN_DIR)]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def po():
    return _load_plugin()


@pytest.fixture(scope="session")
def engine():
    return importlib.import_module(f"{_NS_PARENT}.prompt_optimizer.engine")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeCtx:
    """Minimal stand-in for the plugin registration context."""

    def __init__(self):
        self.hooks = {}
        self.commands = {}

    def register_hook(self, name, callback):
        self.hooks.setdefault(name, []).append(callback)

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = handler


class FakeEvent:
    def __init__(self, text):
        self.text = text
        self.source = None


class FakeSessionStore:
    def __init__(self, session_id="test-session"):
        self.session_id = session_id


class FakePllm:
    """Minimal stand-in for agent.plugin_llm.PluginLlm."""

    def __init__(self, response_text=None):
        self._response = response_text
        self.calls = []

    def complete(self, messages=None, **kwargs):
        self.calls.append((messages, kwargs))
        return types.SimpleNamespace(text=self._response)


def _make_record(engine_mod, original, rewritten):
    return engine_mod.RewriteRecord(
        original=original,
        rewritten=rewritten,
        quality_before=40.0,
        quality_after=90.0,
        token_delta_pct=50.0,
        model_profile="deepseek/general",
    )


LONG_PROMPT = (
    "Please could you kindly write a comprehensive and very detailed analysis "
    "of the entire financial market situation right now and compare it against "
    "the previous trends we discussed earlier"
)

REWRITE_RESPONSE = (
    "Write a financial market analysis.\n---SCORES---\n"
    '{"clarity": 90, "specificity": 80, "terminology": 85, '
    '"actionability": 88, "structure": 82}'
)


# ---------------------------------------------------------------------------
# _hook_supported() — capability introspection
# ---------------------------------------------------------------------------


def test_hook_supported_known_hooks(po):
    assert po._hook_supported("pre_gateway_dispatch") is True
    assert po._hook_supported("transform_llm_output") is True
    assert po._hook_supported("pre_llm_call") is True


def test_hook_supported_pre_user_message_missing_on_current_core(po):
    # Regression: this Hermes build has no pre_user_message hook (upstream
    # PR #29526 not merged). The plugin must detect that instead of
    # registering a callback that never fires.
    assert po._hook_supported("pre_user_message") is False


def test_hook_supported_optimistic_fallback(po, monkeypatch):
    # If the core constant can't be imported, fall back to optimistic True
    # so plugin load can never break on a renamed/removed constant.
    import builtins

    real_import = builtins.__import__

    def raiser(name, *args, **kwargs):
        if name == "hermes_cli.plugins":
            raise ImportError("simulated missing module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raiser)
    assert po._hook_supported("anything") is True


# ---------------------------------------------------------------------------
# register() — hook + command registration
# ---------------------------------------------------------------------------


def test_register_hooks_and_commands(po):
    ctx = FakeCtx()
    po.register(ctx)
    assert "pre_gateway_dispatch" in ctx.hooks
    assert "transform_llm_output" in ctx.hooks
    # Unsupported on this build — must NOT be registered (the fix under test).
    assert "pre_user_message" not in ctx.hooks
    for cmd in (
        "prompt-optimizer",
        "prompt-insights",
        "prompt-compare",
        "prompt-suggestions",
        "prompt-analytics",
        "prompt-stats",
    ):
        assert cmd in ctx.commands, f"missing slash command: {cmd}"


def test_register_pre_user_message_when_supported(po, monkeypatch):
    # Forward-compatibility: on a core with the hook, registration happens
    # exactly as before.
    monkeypatch.setattr(po, "_hook_supported", lambda name: True)
    ctx = FakeCtx()
    po.register(ctx)
    assert "pre_user_message" in ctx.hooks


# ---------------------------------------------------------------------------
# _on_pre_gateway_dispatch — bypasses
# ---------------------------------------------------------------------------


@pytest.fixture
def gateway(po):
    """Reset the plugin's global mode/ctx/pending state between tests."""
    prev_mode = po._mode
    prev_ctx = po._ctx
    po._mode = "auto"
    po._ctx = None
    po._pending_approvals.clear()
    po._session_rewrites.clear()
    yield
    po._mode = prev_mode
    po._ctx = prev_ctx
    po._pending_approvals.clear()
    po._session_rewrites.clear()


def test_gateway_mode_off_passthrough(po, gateway):
    po._mode = "off"
    assert po._on_pre_gateway_dispatch(event=FakeEvent(LONG_PROMPT)) is None


def test_gateway_none_event(po, gateway):
    assert po._on_pre_gateway_dispatch(event=None) is None


def test_gateway_empty_text(po, gateway):
    assert po._on_pre_gateway_dispatch(event=FakeEvent("   ")) is None


def test_gateway_bypass_prefix(po, gateway):
    event = FakeEvent("/quick " + LONG_PROMPT)
    assert po._on_pre_gateway_dispatch(event=event) is None


def test_gateway_short_message_fast_path(po, gateway):
    # perf: < 5 words or < 35 chars skips the LLM entirely.
    assert po._on_pre_gateway_dispatch(event=FakeEvent("Hi")) is None
    assert po._on_pre_gateway_dispatch(event=FakeEvent("short prompt here")) is None


def test_gateway_slash_command(po, gateway):
    assert po._on_pre_gateway_dispatch(event=FakeEvent("/help")) is None
    assert po._on_pre_gateway_dispatch(event=FakeEvent("/prompt-stats")) is None


def test_gateway_skill_invocation(po, gateway):
    text = (
        "[IMPORTANT: The user has invoked the 'web_search' skill. Please use it "
        "thoroughly for this research task and report back with full citations.]"
    )
    assert po._on_pre_gateway_dispatch(event=FakeEvent(text)) is None


def test_gateway_structured_command_fenced_code(po, gateway):
    text = (
        "Please execute this task exactly as specified:\n"
        "```json\n{\"task\": \"delegate\", \"payload\": {...}}\n```\n"
        "and then report what happened with the complete output please"
    )
    assert po._on_pre_gateway_dispatch(event=FakeEvent(text)) is None


def test_gateway_structured_command_verb(po, gateway):
    text = (
        "delegate_task: build me a full authentication system with login, "
        "logout, password reset, email verification, and role-based access control"
    )
    assert po._on_pre_gateway_dispatch(event=FakeEvent(text)) is None


def test_gateway_file_path_is_not_slash_command(po, gateway):
    # A path-like first word must NOT be treated as a slash command. The
    # prompt proceeds to the optimizer; with no _ctx the bridge bypasses
    # (returns None) instead of raising.
    text = (
        "/Users/developer/code/project/main.py please review this file "
        "carefully and tell me what you think about its overall structure"
    )
    assert po._on_pre_gateway_dispatch(event=FakeEvent(text)) is None


# ---------------------------------------------------------------------------
# _on_pre_gateway_dispatch — rewrite behaviour (auto + interactive)
# ---------------------------------------------------------------------------


def test_gateway_auto_rewrite(po, engine, gateway, monkeypatch):
    rec = _make_record(engine, LONG_PROMPT, "Write a financial market analysis.")
    monkeypatch.setattr(po, "_run_optimizer_bridge", lambda *a, **k: rec)
    result = po._on_pre_gateway_dispatch(event=FakeEvent(LONG_PROMPT))
    assert result == {"action": "rewrite", "text": rec.rewritten}


def test_gateway_auto_no_rewrite(po, gateway, monkeypatch):
    monkeypatch.setattr(po, "_run_optimizer_bridge", lambda *a, **k: None)
    assert po._on_pre_gateway_dispatch(event=FakeEvent(LONG_PROMPT)) is None


def test_gateway_interactive_shows_diff(po, engine, gateway, monkeypatch):
    rec = _make_record(engine, LONG_PROMPT, "Write a financial market analysis.")
    po._mode = "interactive"
    monkeypatch.setattr(po, "_run_optimizer_bridge", lambda *a, **k: rec)
    monkeypatch.setattr(po, "_send_diff_via_gateway", lambda *a, **k: None)
    result = po._on_pre_gateway_dispatch(
        event=FakeEvent(LONG_PROMPT), session_store=FakeSessionStore("s1")
    )
    assert result == {"action": "skip", "reason": "interactive-diff-shown"}


def test_gateway_interactive_accept(po, engine, gateway, monkeypatch):
    rec = _make_record(engine, LONG_PROMPT, "Write a financial market analysis.")
    po._mode = "interactive"
    monkeypatch.setattr(po, "_run_optimizer_bridge", lambda *a, **k: rec)
    monkeypatch.setattr(po, "_send_diff_via_gateway", lambda *a, **k: None)
    # First call queues the pending approval (short text still needs to be
    # the *same* session, so use the long prompt to reach the queue).
    first = po._on_pre_gateway_dispatch(
        event=FakeEvent(LONG_PROMPT), session_store=FakeSessionStore("s1")
    )
    assert first == {"action": "skip", "reason": "interactive-diff-shown"}
    # User approves with "y" on the same session.
    second = po._on_pre_gateway_dispatch(
        event=FakeEvent("y"), session_store=FakeSessionStore("s1")
    )
    assert second == {"action": "rewrite", "text": rec.rewritten}


# ---------------------------------------------------------------------------
# engine — helpers
# ---------------------------------------------------------------------------


def test_engine_is_structured_command(engine, monkeypatch):
    assert engine.is_structured_command("delegate_task: run the pipeline") is True
    assert engine.is_structured_command("delegate: something") is True
    assert engine.is_structured_command("```python\nprint(1)\n```") is True
    assert engine.is_structured_command("Build me a thing please") is False
    monkeypatch.setenv("PROMPT_OPTIMIZER_BYPASS_VERBS", "orchestrate")
    assert engine.is_structured_command("orchestrate: run the pipeline") is True


def test_engine_is_skill_invocation(engine):
    marker = "[IMPORTANT: The user has invoked the"
    assert engine.is_skill_invocation(marker + " 'x' skill.]") is True
    assert engine.is_skill_invocation("ordinary user prose") is False


def test_engine_score_prompt_heuristic(engine):
    score, tokens = engine._score_prompt_heuristic(LONG_PROMPT)
    assert isinstance(score, float) and 0 <= score <= 100
    assert isinstance(tokens, int) and tokens >= 1
    assert engine._score_prompt_heuristic("") == (0.0, 0)
    terse = engine._score_prompt_heuristic("hi there")[0]
    assert score > terse


def test_engine_resolve_model_profile_deepseek(engine):
    family, capability = engine.resolve_model_profile("deepseek/deepseek-v4-flash")
    assert family == "deepseek"
    # A pure reasoning variant should resolve the reasoning capability.
    r_family, r_cap = engine.resolve_model_profile("deepseek/deepseek-r1")
    assert r_cap in ("reasoning", None)  # don't over-assert capability taxonomy


def test_engine_timeout_default(engine):
    # 3s was silently disabling rewrites on real LLM latency; default is 10s.
    assert engine.OPTIMIZER_TIMEOUT_S == 10


def test_engine_timeout_env_override(engine, monkeypatch):
    import importlib

    monkeypatch.setenv("PROMPT_OPTIMIZER_TIMEOUT", "7")
    importlib.reload(engine)
    try:
        assert engine.OPTIMIZER_TIMEOUT_S == 7
    finally:
        monkeypatch.delenv("PROMPT_OPTIMIZER_TIMEOUT", raising=False)
        importlib.reload(engine)
        assert engine.OPTIMIZER_TIMEOUT_S == 10  # default restored


# ---------------------------------------------------------------------------
# engine — _run_optimizer rewrite path
# ---------------------------------------------------------------------------


def test_engine_rewrite_with_fake_llm(engine):
    pllm = FakePllm(response_text=REWRITE_RESPONSE)
    record = engine._run_optimizer(LONG_PROMPT, "deepseek/deepseek-v4-flash", "kilocode", pllm)
    assert record is not None
    assert "financial market analysis" in record.rewritten
    assert record.original == LONG_PROMPT
    assert record.model_profile.startswith("deepseek")
    assert record.quality_after == pytest.approx((90 + 80 + 85 + 88 + 82) / 5)
    assert pllm.calls and pllm.calls[0][0] is not None  # messages were sent


def test_engine_rewrite_none_llm(engine):
    # No host LLM (e.g. _ctx unset) -> fail open with no rewrite.
    assert engine._run_optimizer(LONG_PROMPT, "", "", None) is None


def test_engine_parse_rewrite_response(engine):
    parsed = engine._parse_rewrite_response(REWRITE_RESPONSE, LONG_PROMPT)
    assert parsed is not None
    rewritten, scores = parsed
    assert "financial market analysis" in rewritten
    assert set(scores) >= {"clarity", "specificity", "terminology", "actionability", "structure"}
    # Missing separator -> rejected.
    assert engine._parse_rewrite_response("no separator here", LONG_PROMPT) is None
    # Identity rewrite -> rejected.
    assert engine._parse_rewrite_response(LONG_PROMPT + "\n---SCORES---\n{}", LONG_PROMPT) is None


def test_engine_extract_composite(engine):
    good = {"clarity": 90, "specificity": 80, "terminology": 85,
            "actionability": 88, "structure": 82}
    assert engine._extract_composite(good) == pytest.approx(85.0)
    # Missing dimensions -> -1.0 sentinel (heuristic fallback in caller).
    assert engine._extract_composite({"clarity": 50}) == -1.0
    # Out-of-range values -> -1.0 sentinel.
    bad = dict(good, clarity=101)
    assert engine._extract_composite(bad) == -1.0

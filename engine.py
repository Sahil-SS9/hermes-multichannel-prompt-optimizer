"""Shared engine module for prompt-optimizer.

Contains all logic used by both the plugin gateway and CLI hooks:
- Data classes
- Template registry
- Pending preview store (thread-safe)
- Session model cache (thread-safe)
- SQLite metrics store
- Heuristic scoring
- Optimizer engine (LLM rewrite)
- Report formatting and rendering
"""

from __future__ import annotations

import dataclasses
import html
import json
import logging
import os
import re
import sqlite3
import threading
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------------

HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
PLUGIN_DIR = HERMES_HOME / "plugins" / "prompt-optimizer"
METRICS_DB = PLUGIN_DIR / "metrics.db"

OPTIMIZER_TIMEOUT_S = 30
OPTIMIZER_MODEL = "deepseek-v4-flash"

BYPASS_PREFIXES = ("/quick", "*simple", "#basic")

_PENDING_TIMEOUT_S = 120

_LONDON = ZoneInfo("Europe/London") if ZoneInfo else None
_REPORT_DIR = PLUGIN_DIR / "reports"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RewriteRecord:
    original: str
    rewritten: str
    quality_before: float
    quality_after: float
    token_delta_pct: float
    model_profile: str
    approved: bool = True


@dataclass
class PromptTemplate:
    name: str
    description: str
    system_prompt: str


@dataclass
class PromptOptimizationPreview:
    session_key: str
    original: str
    rewritten: str
    quality_before: float
    quality_after: float
    token_delta_pct: float
    model_profile: str
    created_at: float = dataclasses.field(default_factory=time.time)
    template_name: Optional[str] = None


@dataclass
class PromptOptimizationDecision:
    session_key: str
    approved: bool
    applied_text: str
    record: RewriteRecord


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

_TEMPLATES: List[PromptTemplate] = [
    PromptTemplate(
        "default",
        "Balanced clarity and token efficiency",
        "You rewrite user prompts for maximum clarity and token efficiency.\n\n"
        "RULES:\n"
        "1. Preserve ALL intent, constraints, and specific details.\n"
        "2. Remove filler, fluff, and redundant politeness.\n"
        "3. Be direct — start with the action verb.\n"
        "4. Assign quality scores based on the nuance and strength of each dimension.\n\n"
        "OUTPUT FORMAT:\n"
        "Return the optimised prompt first, then a line containing exactly ---SCORES---,\n"
        "then a JSON object with quality assessment per dimension (0-100 each):\n"
        '  - clarity: how unambiguously the intent and desired outcome are expressed\n'
        '  - specificity: how precise the constraints, inputs, and scope are\n'
        '  - terminology: how well domain-specific language is used (or avoided if generic)\n'
        '  - actionability: how clearly the reader knows what to do next\n'
        '  - structure: how well the prompt is organised (sections, lists, ordering)\n\n'
        "ORIGINAL PROMPT:\n{original}\n\n"
        "Rewritten prompt:",
    ),
    PromptTemplate(
        "concise",
        "Maximum brevity without losing intent",
        "You aggressively compress user prompts to their shortest clear form.\n\n"
        "RULES:\n"
        "1. Strip every non-essential word.\n"
        "2. Preserve ONLY the core intent, constraints, and specific values.\n"
        "3. Be direct — start with the action verb.\n\n"
        "OUTPUT FORMAT:\n"
        "Return the optimised prompt first, then a line containing exactly ---SCORES---,\n"
        "then a JSON object with quality assessment per dimension (0-100 each):\n"
        '  - clarity: how unambiguously the intent and desired outcome are expressed\n'
        '  - specificity: how precise the constraints, inputs, and scope are\n'
        '  - terminology: how well domain-specific language is used (or avoided if generic)\n'
        '  - actionability: how clearly the reader knows what to do next\n'
        '  - structure: how well the prompt is organised (sections, lists, ordering)\n\n'
        "ORIGINAL PROMPT:\n{original}\n\n"
        "Rewritten prompt:",
    ),
    PromptTemplate(
        "verbose",
        "Preserve full context while improving structure",
        "You restructure user prompts for maximum clarity while keeping EVERY detail.\n\n"
        "RULES:\n"
        "1. Reorder and rephrase for readability.\n"
        "2. Do NOT remove background, context, or examples.\n"
        "3. Be direct — start with the action verb.\n\n"
        "OUTPUT FORMAT:\n"
        "Return the optimised prompt first, then a line containing exactly ---SCORES---,\n"
        "then a JSON object with quality assessment per dimension (0-100 each):\n"
        '  - clarity: how unambiguously the intent and desired outcome are expressed\n'
        '  - specificity: how precise the constraints, inputs, and scope are\n'
        '  - terminology: how well domain-specific language is used (or avoided if generic)\n'
        '  - actionability: how clearly the reader knows what to do next\n'
        '  - structure: how well the prompt is organised (sections, lists, ordering)\n\n'
        "ORIGINAL PROMPT:\n{original}\n\n"
        "Rewritten prompt:",
    ),
    PromptTemplate(
        "technical",
        "Optimize for code and technical tasks",
        "You rewrite user prompts for maximum clarity on software engineering tasks.\n\n"
        "RULES:\n"
        "1. Preserve ALL file paths, identifiers, stack traces, and constraints.\n"
        "2. Remove filler but keep technical precision.\n"
        "3. Be direct — start with the action verb.\n\n"
        "OUTPUT FORMAT:\n"
        "Return the optimised prompt first, then a line containing exactly ---SCORES---,\n"
        "then a JSON object with quality assessment per dimension (0-100 each):\n"
        '  - clarity: how unambiguously the intent and desired outcome are expressed\n'
        '  - specificity: how precise the constraints, inputs, and scope are\n'
        '  - terminology: how well domain-specific language is used (or avoided if generic)\n'
        '  - actionability: how clearly the reader knows what to do next\n'
        '  - structure: how well the prompt is organised (sections, lists, ordering)\n\n'
        "ORIGINAL PROMPT:\n{original}\n\n"
        "Rewritten prompt:",
    ),
    PromptTemplate(
        "creative",
        "Optimize for creative writing and storytelling",
        "You rewrite user prompts for maximum clarity on creative tasks.\n\n"
        "RULES:\n"
        "1. Preserve tone, style cues, narrative structure, and character details.\n"
        "2. Remove filler and redundant phrasing.\n"
        "3. Be direct — start with the action verb.\n\n"
        "OUTPUT FORMAT:\n"
        "Return the optimised prompt first, then a line containing exactly ---SCORES---,\n"
        "then a JSON object with quality assessment per dimension (0-100 each):\n"
        '  - clarity: how unambiguously the intent and desired outcome are expressed\n'
        '  - specificity: how precise the constraints, inputs, and scope are\n'
        '  - terminology: how well domain-specific language is used (or avoided if generic)\n'
        '  - actionability: how clearly the reader knows what to do next\n'
        '  - structure: how well the prompt is organised (sections, lists, ordering)\n\n'
        "ORIGINAL PROMPT:\n{original}\n\n"
        "Rewritten prompt:",
    ),
]


def select_template(name: str = "") -> PromptTemplate:
    """Return a template by name, falling back to the default."""
    if not name:
        return _TEMPLATES[0]
    for t in _TEMPLATES:
        if t.name == name:
            return t
    return _TEMPLATES[0]


# ---------------------------------------------------------------------------
# Pending preview store (thread-safe)
# ---------------------------------------------------------------------------

_pending_store: Dict[str, PromptOptimizationPreview] = {}
_pending_store_lock = threading.RLock()


def create_preview(session_key: str, preview: PromptOptimizationPreview) -> None:
    with _pending_store_lock:
        _pending_store[session_key] = preview


def resolve_preview(session_key: str) -> Optional[PromptOptimizationPreview]:
    with _pending_store_lock:
        return _pending_store.pop(session_key, None)


def get_pending_preview(session_key: str) -> Optional[PromptOptimizationPreview]:
    with _pending_store_lock:
        return _pending_store.get(session_key)


def pop_and_record(session_key: str) -> Optional[PromptOptimizationPreview]:
    """Pop preview for session_key without side effects."""
    with _pending_store_lock:
        return _pending_store.pop(session_key, None)


def expire_previews(timeout: float = _PENDING_TIMEOUT_S) -> List[str]:
    """Expire stale previews and return the removed session keys."""
    now = time.time()
    expired: List[str] = []
    with _pending_store_lock:
        for key, preview in list(_pending_store.items()):
            if now - preview.created_at > timeout:
                expired.append(key)
        for key in expired:
            del _pending_store[key]
    return expired


# ---------------------------------------------------------------------------
# Session model cache (thread-safe)
# ---------------------------------------------------------------------------

_session_model_cache: Dict[str, str] = {}
_session_model_lock = threading.RLock()


def cache_session_model(session_key: str, model: str) -> None:
    with _session_model_lock:
        _session_model_cache[session_key] = model


def get_cached_session_model(session_key: str) -> Optional[str]:
    with _session_model_lock:
        return _session_model_cache.get(session_key)


# ---------------------------------------------------------------------------
# SQLite metrics store
# ---------------------------------------------------------------------------

_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS rewrites (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    session_id  TEXT,
    platform    TEXT,
    original    TEXT NOT NULL,
    rewritten   TEXT NOT NULL,
    quality_before REAL,
    quality_after  REAL,
    token_delta_pct REAL,
    model_profile TEXT,
    model_used    TEXT,
    mode          TEXT,
    approved      INTEGER DEFAULT 1,
    bypassed      INTEGER DEFAULT 0,
    template_name TEXT,
    strategy      TEXT,
    heuristic_fallback INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_rewrites_ts ON rewrites(ts);
CREATE INDEX IF NOT EXISTS idx_rewrites_session ON rewrites(session_id);
CREATE INDEX IF NOT EXISTS idx_rewrites_model ON rewrites(model_used);
CREATE INDEX IF NOT EXISTS idx_rewrites_profile ON rewrites(model_profile);

CREATE TABLE IF NOT EXISTS daily_stats (
    day         TEXT PRIMARY KEY,
    rewrites    INTEGER DEFAULT 0,
    approved    INTEGER DEFAULT 0,
    bypassed    INTEGER DEFAULT 0,
    avg_quality_before REAL,
    avg_quality_after  REAL,
    avg_token_delta_pct REAL,
    top_profile TEXT
);
"""


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add new columns to existing databases."""
    cursor = conn.execute("PRAGMA table_info(rewrites)")
    existing = {row[1] for row in cursor.fetchall()}
    migrations = {
        "template_name": "ALTER TABLE rewrites ADD COLUMN template_name TEXT",
        "strategy": "ALTER TABLE rewrites ADD COLUMN strategy TEXT",
        "heuristic_fallback": "ALTER TABLE rewrites ADD COLUMN heuristic_fallback INTEGER DEFAULT 0",
    }
    for col, sql in migrations.items():
        if col not in existing:
            try:
                conn.execute(sql)
            except Exception:
                pass
    conn.commit()


def _ensure_db() -> sqlite3.Connection:
    PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(METRICS_DB), check_same_thread=False)
    conn.executescript(_METRICS_SCHEMA)
    _migrate_schema(conn)
    conn.commit()
    return conn


def _db() -> sqlite3.Connection:
    return _ensure_db()


def _prune_old(days: int = 90) -> None:
    cutoff = time.time() - days * 86400
    conn = _db()
    try:
        conn.execute("DELETE FROM rewrites WHERE ts < ?", (cutoff,))
        conn.commit()
    finally:
        conn.close()


def _record(session_id, platform, original, rewritten, quality_before,
            quality_after, token_delta_pct, model_profile, model_used,
            mode, approved=True, bypassed=False, template_name=None,
            strategy=None, heuristic_fallback=False):
    conn = _db()
    try:
        conn.execute(
            """INSERT INTO rewrites
            (ts, session_id, platform, original, rewritten,
             quality_before, quality_after, token_delta_pct,
             model_profile, model_used, mode, approved, bypassed,
             template_name, strategy, heuristic_fallback)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), session_id, platform, original, rewritten,
             quality_before, quality_after, token_delta_pct,
             model_profile, model_used, mode,
             1 if approved else 0, 1 if bypassed else 0,
             template_name, strategy,
             1 if heuristic_fallback else 0))
        conn.commit()
    finally:
        conn.close()
    if hash(str(time.time())) % 100 == 0:
        _prune_old()


def _fetch_rewrites(limit: int = 20, since: float = 0.0, approved_only: bool = False) -> List[Dict[str, Any]]:
    conn = _db()
    conn.row_factory = sqlite3.Row
    try:
        where = ["ts >= ?"]
        params: List[Any] = [since]
        if approved_only:
            where.append("approved = 1")
        sql = (
            "SELECT id, ts, session_id, platform, original, rewritten, "
            "quality_before, quality_after, token_delta_pct, model_profile, model_used, mode, approved, bypassed, "
            "template_name, strategy, heuristic_fallback "
            "FROM rewrites WHERE " + " AND ".join(where) + " ORDER BY ts DESC LIMIT ?"
        )
        params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _summary_for_period(period: str) -> Dict[str, Any]:
    since = _period_start(period)
    conn = _db()
    try:
        row = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN approved=1 THEN 1 ELSE 0 END), "
            "AVG(quality_before), AVG(quality_after), AVG(token_delta_pct) "
            "FROM rewrites WHERE ts >= ?",
            (since,),
        ).fetchone()
        count, approved, qb, qa, td = row or (0, 0, 0, 0, 0)
        return {
            "period": period,
            "count": int(count or 0),
            "approved": int(approved or 0),
            "quality_before": float(qb or 0),
            "quality_after": float(qa or 0),
            "quality_delta": float((qa or 0) - (qb or 0)),
            "token_delta_pct": float(td or 0),
        }
    finally:
        conn.close()


def _analytics_rows(kind: str) -> List[Tuple[str, int, float, float, float]]:
    kind = (kind or "all").lower()
    conn = _db()
    try:
        if kind in {"daily", "day", "today"}:
            expr = "date(ts, 'unixepoch', 'localtime')"
            since = _now_ts() - 14 * 86400
            limit = 14
        elif kind in {"monthly", "month"}:
            expr = "strftime('%Y-%m', ts, 'unixepoch', 'localtime')"
            since = _now_ts() - 366 * 86400
            limit = 12
        else:
            expr = "strftime('%Y-W%W', ts, 'unixepoch', 'localtime')"
            since = _now_ts() - 10 * 7 * 86400
            limit = 10
        rows = conn.execute(
            f"SELECT {expr} AS bucket, COUNT(*), AVG(quality_before), AVG(quality_after), AVG(token_delta_pct) "
            "FROM rewrites WHERE ts >= ? GROUP BY bucket ORDER BY bucket DESC LIMIT ?",
            (since, limit),
        ).fetchall()
        return [(str(r[0]), int(r[1] or 0), float(r[2] or 0), float(r[3] or 0), float(r[4] or 0)) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Scoring (heuristic fallback)
# ---------------------------------------------------------------------------

def _score_prompt_heuristic(text: str) -> Tuple[float, int]:
    """Return (quality_score_0_100, estimated_tokens) via heuristic."""
    if not text:
        return 0.0, 0
    est_tokens = max(1, len(text) // 4)
    score = 0.0
    # 1. Clarity
    action_verbs = {"write", "build", "fix", "refactor", "add", "remove", "explain",
                    "compare", "review", "test", "debug", "create", "generate",
                    "list", "find", "search", "update", "delete"}
    first_ten = " ".join(text.split()[:10]).lower()
    if any(v in first_ten for v in action_verbs):
        score += 20.0
    # 2. Structure
    if any(m in text for m in ("```", "\n- ", "1. ", "##", "<", "| ")):
        score += 20.0
    # 3. Specificity
    if re.search(r"\d+|\.[a-zA-Z]{2,4}|@[a-zA-Z]|\bfile\b|\bpath\b|\bfunction\b", text):
        score += 20.0
    # 4. Conciseness
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]
    if sentences:
        avg_len = est_tokens / max(1, len(sentences))
        if 8 <= avg_len <= 15:
            score += 20.0
        elif avg_len < 25:
            score += 10.0
    # 5. Context
    if any(m in text.lower() for m in ("previous", "earlier", "as we", "continue")):
        score += 20.0
    return min(100.0, score), est_tokens


# ---------------------------------------------------------------------------
# Optimizer engine
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = """You rewrite user prompts for maximum clarity and token efficiency.

RULES:
1. Preserve ALL intent, constraints, and specific details.
2. Remove filler, fluff, and redundant politeness.
3. Be direct — start with the action verb.
4. Assign quality scores based on the nuance and strength of each dimension.

OUTPUT FORMAT:
Return the optimised prompt first, then a line containing exactly ---SCORES---,
then a JSON object with quality assessment per dimension (0-100 each):
  - clarity: how unambiguously the intent and desired outcome are expressed
  - specificity: how precise the constraints, inputs, and scope are
  - terminology: how well domain-specific language is used (or avoided if generic)
  - actionability: how clearly the reader knows what to do next
  - structure: how well the prompt is organised (sections, lists, ordering)

ORIGINAL PROMPT:
{original}

Rewritten prompt:"""


def _parse_rewrite_response(raw: str, original: str) -> Optional[Tuple[str, Dict]]:
    """Parse LLM response into (rewritten_prompt, scores_dict) or None."""
    if not raw:
        return None
    if "---SCORES---" not in raw:
        return None
    parts = raw.split("---SCORES---", 1)
    rewritten = parts[0].strip()
    json_part = parts[1].strip() if len(parts) > 1 else ""

    json_part = re.sub(r"^```(?:json)?\s*", "", json_part).strip()
    json_part = re.sub(r"\s*```$", "", json_part).strip()

    if not rewritten or not json_part:
        return None
    if rewritten.lower() == original.lower():
        return None

    try:
        scores = json.loads(json_part)
    except (json.JSONDecodeError, ValueError):
        return None

    return rewritten, scores


def _extract_composite(scores: Dict) -> float:
    """Extract composite 0-100 score from LLM quality scores.

    Validates that all 5 dimensions are present and 0-100.
    Returns -1.0 sentinel on bad/partial data (triggers heuristic fallback).
    """
    required = {"clarity", "specificity", "terminology", "actionability", "structure"}
    if not required.issubset(scores.keys()):
        return -1.0
    for key in required:
        val = scores[key]
        if not isinstance(val, (int, float)) or val < 0 or val > 100:
            return -1.0
    return sum(scores[k] for k in required) / len(required)


def _try_rewrite_sync(text: str, pllm: Any,
                       template: Optional[PromptTemplate] = None) -> Optional[Tuple[str, float]]:
    """Call pllm.complete to rewrite the user prompt + get quality scores.

    Returns (rewritten_text, composite_score) on success, or None on failure.
    Composite score is 0-100 from LLM quality assessment; -1.0 sentinel when JSON parsed
    but scores were invalid (caller should use heuristic fallback).
    """
    if pllm is None:
        logger.debug("prompt-optimizer: pllm not set — bypassing")
        return None

    tpl = template or select_template()
    user_msg = tpl.system_prompt.format(original=text)

    try:
        result = pllm.complete(
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=512,
            timeout=OPTIMIZER_TIMEOUT_S,
        )
        raw = result.text.strip() if result and result.text else None
        if not raw:
            return None

        parsed = _parse_rewrite_response(raw, text)
        if parsed is None:
            return None

        rewritten, scores = parsed
        composite = _extract_composite(scores)
        return rewritten, composite

    except Exception as exc:
        logger.info("prompt-optimizer: LLM call failed — %s", exc)
        return None


def _run_optimizer(original, model, provider, pllm: Any,
                   template: Optional[PromptTemplate] = None) -> Optional[RewriteRecord]:
    """Run the optimizer: heuristic score before, LLM rewrite + score after."""
    quality_before, tokens_before = _score_prompt_heuristic(original)
    logger.info("prompt-optimizer: trying rewrite for: %r", original[:80])

    result = _try_rewrite_sync(original, pllm, template=template)
    if not result:
        logger.info("prompt-optimizer: no rewrite produced")
        return None

    rewritten_text, llm_composite = result

    if llm_composite >= 0:
        quality_after = llm_composite
    else:
        quality_after, _ = _score_prompt_heuristic(rewritten_text)

    tokens_after = max(1, len(rewritten_text) // 4)
    token_delta_pct = round(
        ((tokens_before - tokens_after) / max(1, tokens_before)) * 100, 1)

    return RewriteRecord(
        original=original, rewritten=rewritten_text,
        quality_before=quality_before, quality_after=quality_after,
        token_delta_pct=token_delta_pct,
        model_profile=model or "unknown",
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_diff(record):
    """Compact diff for interactive approval.

    Shows original→rewritten quality scores, token delta, and both
    versions clipped for display.
    """
    if record.token_delta_pct > 0:
        token_msg = f"saved -{abs(record.token_delta_pct):.0f}%"
    elif record.token_delta_pct < 0:
        token_msg = f"used +{abs(record.token_delta_pct):.0f}%"
    else:
        token_msg = "no token change"

    orig = _clip(record.original, 200)
    rwt = _clip(record.rewritten, 200)
    q_diff = record.quality_after - record.quality_before

    return (
        f"Prompt Optimizer — Diff\n"
        f"Quality: {record.quality_before:.0f} \N{RIGHTWARDS ARROW} "
        f"{record.quality_after:.0f} ({q_diff:+.0f})  |  "
        f"Tokens: {token_msg}\n\n"
        f"[Original]\n{orig}\n\n"
        f"[Rewritten]\n{rwt}\n\n"
        f"Reply Y to use  |  N to keep original"
    )


def _coerce_text(value: Any) -> str:
    """Best-effort conversion for plugin inputs that sometimes arrive as tuples/lists."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_coerce_text(v) for v in value if v is not None).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "message", "value"):
            if key in value:
                return _coerce_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# ---------------------------------------------------------------------------
# Report utilities
# ---------------------------------------------------------------------------

def _normalise_args(raw_args: Any) -> List[str]:
    text = _coerce_text(raw_args).strip()
    return text.split() if text else []


def _now_ts() -> float:
    return time.time()


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(float(ts or 0), tz=_LONDON)


def _fmt_ts(ts: float) -> str:
    return _dt(ts).strftime("%d/%m/%y %H:%M:%S")


def _fmt_day(ts: float) -> str:
    return _dt(ts).strftime("%d/%m/%y")


def _period_start(period: str) -> float:
    now = datetime.now(tz=_LONDON)
    period = (period or "week").lower()
    if period in {"today", "day", "daily"}:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period in {"month", "monthly"}:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif period in {"all", "total"}:
        return 0.0
    else:
        start = now - timedelta(days=7)
    return start.timestamp()


def _limit_from_args(argv: Sequence[str], default: int = 5, maximum: int = 20) -> int:
    for idx, arg in enumerate(argv):
        if arg == "--limit" and idx + 1 < len(argv):
            try:
                return max(1, min(maximum, int(argv[idx + 1])))
            except ValueError:
                return default
        if arg.startswith("--limit="):
            try:
                return max(1, min(maximum, int(arg.split("=", 1)[1])))
            except ValueError:
                return default
    return default


def _delta_words(delta: float) -> str:
    if delta > 0:
        return f"saved {delta:.1f}%"
    if delta < 0:
        return f"expanded {abs(delta):.1f}%"
    return "flat"


def _quality_delta(row: Dict[str, Any]) -> float:
    return float(row.get("quality_after") or 0) - float(row.get("quality_before") or 0)


def _clip(text: Any, width: int = 110) -> str:
    value = " ".join(_coerce_text(text).split())
    if len(value) <= width:
        return value
    return value[: max(0, width - 1)].rstrip() + "…"


def _wrap_block(text: Any, indent: str = "  ", width: int = 96) -> List[str]:
    value = _coerce_text(text).strip()
    if not value:
        return [indent + "(empty)"]
    lines: List[str] = []
    for paragraph in value.splitlines() or [value]:
        wrapped = textwrap.wrap(paragraph, width=width) or [""]
        lines.extend(indent + line for line in wrapped)
    return lines


def _bar(before: float, after: float, width: int = 16) -> str:
    before = max(0, min(100, float(before or 0)))
    after = max(0, min(100, float(after or 0)))
    filled = int(round(after / 100 * width))
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _render_header(title: str, subtitle: str = "") -> List[str]:
    stamp = datetime.now(tz=_LONDON).strftime("%d/%m/%y %H:%M:%S")
    lines = [f"{title} · {stamp}"]
    if subtitle:
        lines.append(subtitle)
    lines.append("─" * max(64, min(100, len(lines[0]))))
    return lines


def _render_summary_cards() -> List[str]:
    today = _summary_for_period("daily")
    week = _summary_for_period("weekly")
    month = _summary_for_period("monthly")
    lines = ["Overview"]
    lines.append("  Period       Rewrites  Quality      Tokens")
    for label, data in (("Today", today), ("7 days", week), ("Month", month)):
        q = f"{data['quality_before']:.0f}→{data['quality_after']:.0f} ({data['quality_delta']:+.0f})"
        lines.append(f"  {label:<11} {data['count']:>7}  {q:<12} {_delta_words(data['token_delta_pct'])}")
    return lines


def _render_comparisons(rows: Sequence[Dict[str, Any]], *, title: str = "Before / after comparisons") -> List[str]:
    lines = [title]
    if not rows:
        return lines + ["  No rewrites recorded yet."]
    for idx, row in enumerate(rows, 1):
        q_before = float(row.get("quality_before") or 0)
        q_after = float(row.get("quality_after") or 0)
        lines.append("")
        lines.append(
            f"  #{idx} id={row.get('id')} · {_fmt_ts(row.get('ts') or 0)} · "
            f"{row.get('platform') or 'unknown'} · quality {q_before:.0f}→{q_after:.0f} "
            f"({_quality_delta(row):+.0f}) · tokens {_delta_words(float(row.get('token_delta_pct') or 0))}"
        )
        lines.append(f"  {_bar(q_before, q_after)}")
        lines.append("  Before")
        lines.extend(_wrap_block(row.get("original"), indent="    "))
        lines.append("  After")
        lines.extend(_wrap_block(row.get("rewritten"), indent="    "))
    return lines


def _render_suggestion_rows(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = ["Prompt replacement suggestions"]
    if not rows:
        return lines + ["  No suggestions yet. Send a few prompts first."]
    for idx, row in enumerate(rows, 1):
        gain = _quality_delta(row)
        lines.append("")
        lines.append(f"  {idx}. Replace this pattern · quality {gain:+.0f} · tokens {_delta_words(float(row.get('token_delta_pct') or 0))}")
        lines.append(f"     Instead of: {_clip(row.get('original'), 100)}")
        lines.append(f"     Use:        {_clip(row.get('rewritten'), 100)}")
    return lines


def _render_insight_bullets(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = ["Optimization insights"]
    if not rows:
        return lines + ["  No data yet."]
    expanded = [r for r in rows if float(r.get("token_delta_pct") or 0) < 0]
    saved = [r for r in rows if float(r.get("token_delta_pct") or 0) > 0]
    avg_gain = sum(_quality_delta(r) for r in rows) / max(1, len(rows))
    avg_tokens = sum(float(r.get("token_delta_pct") or 0) for r in rows) / max(1, len(rows))
    lines.append(f"  • Quality is improving by {avg_gain:+.1f} points on average across recent prompts.")
    if avg_tokens < 0:
        lines.append("  • The optimiser is often expanding prompts. That can improve clarity, but it is not pure token saving.")
    else:
        lines.append("  • The optimiser is reducing token load while preserving stronger structure.")
    if expanded:
        lines.append(f"  • {len(expanded)}/{len(rows)} recent rewrites expanded the prompt. Watch for over-engineered short commands.")
    if saved:
        best = max(saved, key=lambda r: float(r.get("token_delta_pct") or 0))
        lines.append(f"  • Best token save: {_delta_words(float(best.get('token_delta_pct') or 0))} on id={best.get('id')}.")
    top = max(rows, key=_quality_delta)
    lines.append(f"  • Best quality gain: {_quality_delta(top):+.0f} on id={top.get('id')}.")
    lines.append("  • Use /quick, *simple, or #basic to bypass optimisation for tiny commands.")
    return lines


def _display_bucket(bucket: str, mode: str) -> str:
    try:
        if mode in {"daily", "day", "today"}:
            return datetime.strptime(bucket, "%Y-%m-%d").strftime("%d/%m/%y")
        if mode in {"monthly", "month"}:
            return datetime.strptime(bucket + "-01", "%Y-%m-%d").strftime("01/%m/%y")
        if "-W" in bucket:
            year, week = bucket.split("-W", 1)
            start = datetime.strptime(f"{year} {int(week)} 1", "%Y %W %w")
            return start.strftime("%d/%m/%y")
    except Exception:
        pass
    return bucket


def _render_analytics(kind: str = "all") -> str:
    argv_kind = (kind or "all").lower()
    modes = ["daily", "weekly", "monthly"] if argv_kind in {"", "all"} else [argv_kind]
    lines = _render_header("Prompt Analytics", "Daily, weekly, and monthly optimisation metrics")
    for mode in modes:
        rows = _analytics_rows(mode)
        lines.append("")
        lines.append(mode.capitalize())
        if not rows:
            lines.append("  No data.")
            continue
        lines.append("  Period       Rewrites  Quality      Tokens")
        for bucket, count, qb, qa, td in rows:
            label = _display_bucket(bucket, mode)
            lines.append(f"  {label:<12} {count:>7}  {qb:.0f}→{qa:.0f} ({qa-qb:+.0f})  {_delta_words(td)}")
    return "\n".join(lines)


def _build_full_report(limit: int = 5) -> str:
    recent = _fetch_rewrites(limit=max(limit, 10), since=0, approved_only=False)
    suggestions = sorted(recent, key=_quality_delta, reverse=True)[:limit]
    lines = _render_header(
        "Prompt Optimizer Insights",
        "Backend-wired prompt rewrites surfaced as a CLI/TUI report",
    )
    lines.extend(_render_summary_cards())
    lines.append("")
    lines.extend(_render_insight_bullets(recent[:10]))
    lines.append("")
    lines.extend(_render_suggestion_rows(suggestions))
    lines.append("")
    lines.extend(_render_comparisons(recent[:limit]))
    lines.append("")
    lines.append("Discoverability")
    lines.append("  /prompt-insights [--html]       Full report")
    lines.append("  /prompt-compare [--limit N]     Before/after comparisons")
    lines.append("  /prompt-suggestions [--limit N] Replacement suggestions")
    lines.append("  /prompt-analytics [all|daily|weekly|monthly]")
    return "\n".join(lines)


def _write_html_report(report_text: str) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=_LONDON).strftime("%Y-%m-%d-%H%M%S")
    path = _REPORT_DIR / f"report-{stamp}.html"
    body = html.escape(report_text)
    doc = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>Prompt Optimizer Insights</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; background: #0f1115; color: #e8e6df; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  main {{ max-width: 1120px; margin: 0 auto; padding: 32px; }}
  pre {{ white-space: pre-wrap; line-height: 1.45; background: #171a21; border: 1px solid #2d3340; border-radius: 16px; padding: 24px; box-shadow: 0 18px 60px rgba(0,0,0,.35); }}
  .tag {{ color: #fbbf24; margin-bottom: 16px; }}
</style>
</head>
<body><main><div class=\"tag\">Prompt Optimizer Insights</div><pre>{body}</pre></main></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return str(path)

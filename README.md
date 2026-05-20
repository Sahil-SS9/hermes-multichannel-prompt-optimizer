# prompt-optimizer

Model-aware prompt rewriter with token-efficiency scoring, coaching metrics, and slash commands.

## What it does

- **Intercepts** every user message before the agent sees it (gateway + CLI/TUI)
- **Rewrites** it for the target model's strengths (Claude XML, DeepSeek reasoning, OpenAI function calling, Gemini long-context, reasoning models)
- **Scores** before/after prompt quality using LLM-based assessment with heuristic fallback
- **Stores** metrics in SQLite for longitudinal coaching
- **Badges** each response: `Optimized · +12% tokens · quality 45→72 · /prompt-stats`
- **Bypass** with `/quick`, `*simple`, or `#basic` prefixes

## Modes

| Mode | Behaviour |
|------|-----------|
| `auto` (default) | Silent rewrite — agent sees optimized version, you see badge |
| `interactive` | Show diff → approve (y) or reject (n) before rewrite |
| `off` | Pass through untouched |

### Interactive mode details

- **Gateway (Telegram/Discord)**: Diff sent as a reply. Reply `y` to use optimized, `n` to use original. 
- **CLI/TUI**: Diff printed to terminal. Type `y`/`n` at the prompt.
- **Timeout**: Pending approval expires after 120s — auto-approved, new message gets a fresh diff.
- **Rapid-fire**: If you send a new message before responding `y/n`, the pending rewrite is auto-approved and the new message is processed.
- **Non-TTY**: Falls back to auto mode.

## Scoring

The optimizer LLM evaluates each prompt across 5 dimensions (0-100 each):
- **clarity** — action verb early, clear intent
- **specificity** — concrete nouns, file paths, numbers
- **terminology** — domain-correct terms, no vague language
- **actionability** — can the agent act on this without clarification?
- **structure** — formatting appropriate for the target model

Composite score is the average. Falls back to heuristic if LLM call fails or times out.

## Commands

| Command | Description |
|---------|-------------|
| `/prompt-optimizer auto` | Silent rewrite (default) |
| `/prompt-optimizer interactive` | Show diff, ask approval |
| `/prompt-optimizer off` | Pass through untouched |
| `/prompt-optimizer status` | Current mode, stored rewrites, report commands |
| `/prompt-insights` | Full CLI/TUI report: overview, insights, suggestions, comparisons, analytics |
| `/prompt-insights --html` | Full report plus a local `file://` HTML report under `reports/` |
| `/prompt-compare --limit 5` | Latest before/after prompt comparisons |
| `/prompt-suggestions --limit 8` | Reusable replacement suggestions from recent rewrites |
| `/prompt-analytics all` | Daily, weekly, and monthly analytics |
| `/prompt-analytics daily` | Daily analytics only |
| `/prompt-analytics weekly` | Weekly analytics only |
| `/prompt-analytics monthly` | Monthly analytics only |
| `/prompt-stats` | Legacy compact alias for `/prompt-insights` |
| `/prompt-stats --raw` | JSON summary for today/week/month |

## Files

- `__init__.py` — plugin logic (hooks, slash commands, metrics DB)
- `model-profiles.yaml` — per-model optimization strategies (user-editable)
- `metrics.db` — SQLite store (auto-created, 90-day rolling prune)

## Hooks used

- `pre_gateway_dispatch` — rewrite incoming gateway messages
- `pre_user_message` — rewrite CLI/TUI messages
- `transform_llm_output` — append badge after assistant responses

## Config

No `config.yaml` keys required. The plugin reads:
- `plugins.entries.prompt-optimizer.llm.*` for LLM override trust (standard Hermes)

## Author

KENSEI / Octacon

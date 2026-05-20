# hermes-multichannel-prompt-optimizer

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that rewrites your prompts before they reach the LLM. Works across every surface Hermes runs on: CLI, TUI, Discord, Telegram, and any other gateway adapter.

Same agent, sharper prompts, lower token bills, better answers — without you having to think about prompt craft.

---

## Why

A senior PM types `"hey could you maybe explain to me very kindly what python generators are please when you get a chance"` — 18 words, mostly filler. The model burns context on politeness and noise. With this plugin, the agent sees `"Explain Python generators."` — clearer, cheaper, and the answer comes back sharper.

This pattern repeats across every conversation. Over a month, the savings are meaningful: in dogfooding so far the optimiser averages **+55 quality points** and **20–80% token reduction** per rewrite. You also build up a private dataset of your own prompt patterns and improvements, viewable via `/prompt-insights`.

---

## What it does

- **Intercepts** every user message before it reaches the agent — on CLI, TUI, and any gateway platform.
- **Rewrites** for clarity, specificity, and token efficiency using a fast secondary model (defaults work, configurable).
- **Scores** before/after across 5 dimensions: clarity, specificity, terminology, actionability, structure.
- **Records** every rewrite into a local SQLite database for analytics and longitudinal coaching.
- **Surfaces** insights via slash commands: comparisons, reusable suggestions, analytics by day/week/month.
- **Renders** an arrow-key approval overlay on CLI (via `ctx.ask_user`) and a full before/after panel in the TUI.

---

## Surfaces

| Surface | Auto mode | Interactive mode |
|---|---|---|
| `hermes chat` (CLI) | Silent rewrite | Arrow-key overlay (accept / reject) |
| `hermes chat --tui` | Silent rewrite | Before/after panel with quality scores |
| Discord | Silent rewrite | Diff sent as a message; reply `y` / `n` |
| Telegram, Slack, IRC, etc. | Silent rewrite | Same as Discord |

---

## Requirements

- **Hermes Agent** with the `pre_user_message` plugin hook. This hook is required for CLI/TUI rewrites to work. If your Hermes build is missing it, the gateway path (Discord/Telegram/…) still works via `pre_gateway_dispatch`.
- Python 3.11+
- An LLM provider configured in Hermes for the optimiser model (the plugin uses Hermes's `ctx.llm` facade, so it inherits your active provider/auth — no separate keys needed by default).

---

## Install

```bash
hermes plugins install Sahil-SS9/hermes-multichannel-prompt-optimizer
```

Then enable it in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - prompt-optimizer
```

Restart your Hermes session. Confirm it's loaded:

```bash
hermes plugins list | grep prompt-optimizer
```

You should see it as `enabled`. Then in `hermes chat`:

```
/prompt-optimizer status
```

If the status block prints, you're set.

---

## Modes

| Mode | Behaviour |
|---|---|
| `auto` *(default)* | Silent rewrite — agent sees the optimised version, you don't see the diff. |
| `interactive` | Show the diff first, ask for approval before sending. |
| `off` | Pass everything through untouched. |

Toggle mid-session:

```
/prompt-optimizer auto
/prompt-optimizer interactive
/prompt-optimizer off
```

---

## Slash commands

| Command | Description |
|---|---|
| `/prompt-optimizer [auto\|interactive\|off\|status]` | Set mode or print status. |
| `/prompt-insights` | Full report: overview, insights, suggestions, comparisons, analytics. |
| `/prompt-insights --html` | Same report plus a styled HTML file under `reports/`. |
| `/prompt-compare --limit 5` | Latest before/after comparisons. |
| `/prompt-suggestions --limit 8` | Reusable prompt-replacement patterns mined from your history. |
| `/prompt-analytics [daily\|weekly\|monthly\|all]` | Period analytics. |
| `/prompt-stats --raw` | JSON summary for today, week, month. Useful for cron / dashboards. |

---

## Configuration

The plugin needs no `config.yaml` entries to run with sensible defaults. To pin the optimiser to a specific cheap-and-fast model, override under `plugins.entries`:

```yaml
plugins:
  enabled:
    - prompt-optimizer
  entries:
    prompt-optimizer:
      llm:
        allow_model_override: true
        allowed_models:
          - deepseek-v4-flash
        allow_provider_override: true
        allowed_providers:
          - nous
```

This isolates the optimiser's LLM cost from your main session model — you can run Claude Opus for the agent while a £0.05/M token model handles rewrites.

### Model profiles

`model-profiles.yaml` ships with prompt templates tuned per target model family (Claude XML, OpenAI function-calling, reasoning models, long-context, etc.). Edit it to customise rewrite strategy per model. The defaults are reasonable for most users.

---

## Privacy

- All metrics live in a **local SQLite database** at `~/.hermes/plugins/prompt-optimizer/metrics.db`. Nothing is uploaded.
- The optimiser does call your configured LLM provider for the rewrite step — that's a third-party API call subject to your provider's privacy policy. If you don't want any external calls, set `/prompt-optimizer off`.
- The local database keeps 90 days of rewrites by default before pruning. Delete `metrics.db` any time to reset.

---

## Hooks used

| Hook | Purpose |
|---|---|
| `pre_user_message` | Rewrite messages from CLI / TUI before they reach the agent. |
| `pre_gateway_dispatch` | Rewrite messages from Discord / Telegram / Slack / etc. |
| `transform_llm_output` | Append an inline quality-badge to the assistant's reply when a rewrite happened. |

---

## Bypass prefixes

The plugin used to support `/quick`, `*simple`, `#basic` as one-off bypasses. In practice the slash-command dispatcher in `hermes chat` claims anything starting with `/`, so only gateway surfaces honour the prefixes reliably. **Recommended**: use mode flips (`/prompt-optimizer off` then `/prompt-optimizer auto`) instead.

---

## Development

Clone, edit, link into Hermes:

```bash
git clone https://github.com/Sahil-SS9/hermes-multichannel-prompt-optimizer ~/.hermes/plugins/prompt-optimizer
hermes plugins enable prompt-optimizer
```

Run the test suite:

```bash
cd /path/to/hermes-agent
venv/bin/pytest tests/plugins/test_prompt_optimizer_plugin.py -v
```

PRs welcome. Please include tests for any new hook semantics or scoring changes.

---

## Roadmap

- [ ] Wire `model_used` field for CLI rewrites (currently hardcoded blank — minor analytics gap).
- [ ] Per-user model-profile overrides (currently global).
- [ ] Optional GitHub Actions example for cron-driven weekly digests posted to Discord/Slack.

---

## Credits

Built by [Sahil Saghir](https://github.com/Sahil-SS9) for the KENSEI / Octacon personal-agent stack. Released under MIT in case it's useful to anyone else running Hermes Agent in production.

---

## License

MIT — see [LICENSE](./LICENSE).

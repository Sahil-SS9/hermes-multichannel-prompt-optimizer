# hermes-multichannel-prompt-optimizer (retired)

This plugin is retired. As of 3 October 2026 it is no longer maintained and the repository is archived (read-only).

## Why

The plugin rewrote your prompts before they reached the model, on the theory that a cleaner prompt gets a better answer. A controlled test did not support that.

- 100 real user messages from the WildChat-1M dataset, held back for this test and used once.
- Each message was answered twice: once as written, once after the optimizer rewrote it.
- A judge compared each pair of answers in both orders. It saw the user's original message, never the rewrite.
- claude-sonnet-5-5 did the rewriting, the answering and the judging.
- The main pass condition was fixed before the run: answers to rewritten prompts must be preferred on average, with the 95% confidence interval above zero.

| Version tested | Prompts rewritten | Answers better / worse / tied | Mean preference (95% CI) |
|---|---|---|---|
| Development version | 67 of 100 | 30 / 29 / 41 | -0.010 (-0.155 to 0.140) |
| Rebuilt version | 19 of 100 | 8 / 7 / 85 | -0.005 (-0.085 to 0.070) |

Both versions are later than the code published here, and neither was released. Neither passed. Rewriting made answers worse about as often as it made them better.

The earlier claims in this README of better answers and "+55 quality points" per rewrite are withdrawn. The quality points were the plugin's own score of the prompt text, which does not measure whether the answer improved.

## If you have it installed

```bash
hermes plugins disable prompt-optimizer
hermes plugins remove prompt-optimizer
```

Then restart Hermes, including any running gateway. Sessions that are already open keep the old code in memory until they restart.

`remove` deletes the plugin folder, `plugins/prompt-optimizer` inside your Hermes home (usually `~/.hermes`). That folder also holds `metrics.db`, the local log of your original and rewritten prompts, and any HTML reports, so they go with it. The log was only ever stored on your machine.

## The code

The code stays here, unchanged, under the MIT licence. The previous README, with setup and command docs, is at commit [e83eba9](https://github.com/Sahil-SS9/hermes-multichannel-prompt-optimizer/blob/e83eba9a8e102ed255a5bfd20cc88ea50e3af28f/README.md).

Built by [Sahil Saghir](https://github.com/Sahil-SS9), with contributions from [benzntech](https://github.com/benzntech) ([PR #4](https://github.com/Sahil-SS9/hermes-multichannel-prompt-optimizer/pull/4)).

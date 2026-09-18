# Using pi (deepseek-flash / glm-5.3-flash) in this project

pi is the general-purpose coding agent for well-specified, mechanical work: loaders, flags,
tests for a stated spec, doc paragraphs. It is cheap (two separate quotas: `deepseek` and
`openrouter`), so it does the bulk of the typing; the main session specifies, reviews and
integrates. Opus/Codex are reserved for judgement (design, fidelity to the paper, review).

```bash
pi -p --no-session --provider deepseek   --model deepseek-flash     --thinking medium "$(cat notes/briefs/pi-NN-<slug>.md)"
pi -p --no-session --provider openrouter --model z-ai/glm-5.3-flash --thinking medium "$(cat notes/briefs/pi-NN-<slug>.md)"
```

- Briefs live in `notes/briefs/pi-NN-<slug>.md` and are self-contained: pi has no conversation
  context, so the brief inlines the facts, the pins, the allowed file list, the tests to write
  first, and the gates to run. Two pi runs in parallel must have disjoint file lists.
- Run from the repo root, in the background, output to the scratchpad; never with a trailing `&`.
- pi does not commit. The main session diffs the result, re-runs `sh scripts/check-all.sh`, and
  commits with the brief number in the message.
- Use `deepseek-flash` first; switch to `glm-5.3-flash` when that quota is exhausted (or run
  one brief on each when parallel).

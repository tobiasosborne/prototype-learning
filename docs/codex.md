# Using Codex (gpt-5.6-sol, xhigh) in this project

Codex is used for work that benefits from *independence* from the main session: blind second
implementations used as test oracles, adversarial reviews, and derivations.

```bash
codex exec --skip-git-repo-check -C /home/tobias/Projects/prototype-learning \
  -s workspace-write --json - < notes/briefs/codex-NN-<slug>.md \
  > experiments/runs/codex/NN-events.jsonl 2> experiments/runs/codex/NN-err.log
```

- Briefs live in `notes/briefs/` and are self-contained: Codex has no conversation context, so
  inline every convention (qubit 0 = MSB, CRX matrix, file paths) and the exact deliverable.
- Sandbox: `read-only` for reviews/derivations, `workspace-write` for code that must run.
- Always `--json` to a file; the final message is in the last `agent_message` item.
- Run it in the background (`run_in_background`), never with a trailing `&` inside the tool
  shell — the detached process gets killed when the tool returns.
- Results are integrated only after the main session re-runs the tests itself.

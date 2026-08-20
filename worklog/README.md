# Worklog shard format

Each shard holds consecutive sessions, newest first, separated by `# (earlier: …)` markers.

```
# Session N — YYYY-MM-DD — <headline>
## HANDOFF — state at session close
## What was done
## Verification            (commands run + outcome, verbatim failures)
## Gotchas for future agents
```

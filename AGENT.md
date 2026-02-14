use scratch_pad file inside a local folder named '.local/[agent_name]_scratch_pad.md' where you will write in your mistakes and/or wrong assumptions and how you fix them. If '.local/[agent_name]_scratch_pad.md' doesn't exist creat it and make sure the ".local" folder in gitignore

## Kernel Generation Process

IMPORTANT: read this file now 'skills/gpu-kernel-engineering.md' it has all you need to be world class at kernel engineering

## Experiment Logging

Agents must use `experiment-log.md` to record each meaningful experiment iteration, including hypothesis, action, outcome, and breakthrough insight.

## Scratch Pad Entry Template

Use this format for each mistake logged in `.local/[agent_name]_scratch_pad.md`:

```markdown
## YYYY-MM-DD HH:MM - <short title>
- Assumption/Mistake: <what was wrong>
- Impact: <what it affected>
- Fix Applied: <what changed>
- Prevention: <how to avoid repeating it>
```

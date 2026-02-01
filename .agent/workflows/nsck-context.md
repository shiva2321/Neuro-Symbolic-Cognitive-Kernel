---
description: Get context for NSCK AGI project before starting work
---

# NSCK Project Context Workflow

// turbo-all

## Steps

1. Read the project context document
```
cat d:\NSCK_v1\docs\AGENT_CONTEXT.md
```

2. Read the current task tracker for progress status
```
View the task.md in the brain artifacts folder
```

3. Check the session handoff for any blocking items or decisions
```
View SESSION_HANDOFF.md in the brain artifacts folder
```

4. Ask the user what they want to work on today

## Important Files

- `d:\NSCK_v1\docs\AGENT_CONTEXT.md` - Quick project overview
- `d:\NSCK_v1\nsck-demo\python\python_server.py` - Main brain code
- `d:\NSCK_v1\nsck-demo\python\snn_qat.py` - Neural network
- `d:\NSCK_v1\nsck-demo\rust_vsa\src\lib.rs` - Symbolic engine

## Before Ending Session

1. Update SESSION_HANDOFF.md with:
   - Actions taken
   - Decisions made
   - What next agent should do

2. Update task.md with completed items

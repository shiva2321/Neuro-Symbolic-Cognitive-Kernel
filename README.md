# NSCK AGI Project

## Quick Start for Agents

> **CRITICAL:** Before starting ANY work, read the session handoff document.

### Required Reading Order
1. `docs/AGENT_CONTEXT.md` ← Start here
2. `.gemini/antigravity/brain/.../SESSION_HANDOFF.md`
3. `.gemini/antigravity/brain/.../task.md`
4. `.gemini/antigravity/brain/.../implementation_plan.md`

---

## Project Structure

```
NSCK_v1/
├── nsck-demo/
│   ├── python/           # Main Python code
│   │   ├── python_server.py    # Brain control loop
│   │   ├── snn_qat.py          # Neural network
│   │   ├── metacognition.py    # Arbitration
│   │   └── ...                 # 35+ modules
│   ├── rust_vsa/         # Hypervector engine (Rust)
│   └── tests/            # Test suite
├── docs/
│   └── AGENT_CONTEXT.md  # Quick context for agents
└── requirements.txt
```

## Current Goal

Building sentient AGI prototype with:
- Autonomous learning (no teacher dependency)
- Multi-step reasoning and planning
- Self-awareness and metacognition
- Causal understanding
- Imagination/world models
- Transfer to unseen environments

See implementation_plan.md for 24-week roadmap.

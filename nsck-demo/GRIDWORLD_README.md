# GridWorld Survival - Quick Start Guide

## What Is This?

A **complex game environment** with an **intelligent learning agent** that demonstrates:

✅ **Learning from scratch** - Agent starts with no knowledge and learns strategies  
✅ **Adaptation** - Responds to dynamic enemies and changing situations  
✅ **Multi-objective decisions** - Balances survival, resources, and progress  
✅ **Improvement over time** - Gets better with experience  
✅ **Transfer learning** - Knowledge transfers to other games  
✅ **Comprehensive logging** - Everything recorded for analysis  

## Why This Game?

**More complex than Snake/Maze/Pong:**

| Feature | GridWorld | Basic Games |
|---------|-----------|-------------|
| Objectives | **Multiple** | Single |
| Resources | **4 stats to manage** | None |
| Threats | **Dynamic patrols** | Static |
| Planning | **Required** | Optional |
| State Space | **Large** | Small |

## Quick Demo (30 seconds)

```bash
# See what the game is about
cd nsck-demo/python
python demo_gridworld.py --complexity

# Run a quick demo (no dependencies needed)
python demo_gridworld.py --episodes 3
```

## Train Your Own Agent

```bash
# Full training with logging (requires NSCK setup)
python train_gridworld_agent.py --episodes 50

# Your logs will be saved to: gridworld_logs/session_*/
```

## What Gets Logged?

Every training session creates:

- 📊 **game_log.jsonl** - Every decision, state, reward
- 📈 **metrics.jsonl** - Episode scores, survival times
- 🧠 **learning_events.jsonl** - Rules learned, adaptations
- 📝 **agent.log** - Human-readable training log
- 📄 **summary.txt** - Complete analysis

## Example Log Analysis

```python
import json

# Load a training session
with open('gridworld_logs/session_*/game_log.jsonl') as f:
    steps = [json.loads(line) for line in f]

# See what the agent was thinking
step = steps[100]
print(f"State: {step['state']['predicates']}")
print(f"Action: {step['action']}")
print(f"Reasoning: {step['reasoning']['explanation']}")
print(f"Confidence: {step['reasoning']['confidence']}")
```

## Game Symbols

```
@  = Player (you/agent)
F  = Food (reduces hunger)
W  = Water (reduces thirst)
E  = Enemy (patrols, damages you)
X  = Hazard (static damage)
T  = Treasure (bonus points)
★  = Exit (goal!)
#  = Wall
```

## The Challenge

**Survive and reach the exit while:**
- Managing energy (drains over time)
- Avoiding enemies (they patrol!)
- Collecting resources (food, water)
- Dodging hazards
- Making strategic trade-offs

## Transfer Learning

The agent learns **abstract concepts** that work across games:

```
GridWorld Concept → Abstract → Other Games
------------------------------------------
FOOD, WATER       → RESOURCE → SNAKE_FOOD
ENEMY, HAZARD     → THREAT   → MAZE_WALL
EXIT              → GOAL     → MAZE_EXIT
```

**This means:** Rules learned in GridWorld automatically apply to Snake, Maze, and any new game you add!

## Requirements

**Minimal (for demo):**
- Python 3.11+
- numpy, scipy

**Full (for agent training):**
- Above plus NSCK dependencies
- See main README.md for installation

## Files

```
nsck-demo/python/
├── gridworld_survival.py      # Game environment
├── train_gridworld_agent.py   # Full training system
└── demo_gridworld.py           # Standalone demo

nsck-demo/tests/
└── test_gridworld_agent.py    # Tests

docs/
└── GRIDWORLD_AGENT.md          # Full documentation
```

## Full Documentation

📖 See [docs/GRIDWORLD_AGENT.md](../docs/GRIDWORLD_AGENT.md) for complete documentation including:
- Detailed architecture
- Training results
- Log analysis
- Transfer learning setup
- Troubleshooting

## Quick Test

```bash
# Run tests (if pytest available)
pytest nsck-demo/tests/test_gridworld_agent.py -v

# Or run directly
python nsck-demo/tests/test_gridworld_agent.py
```

## What Makes This Special?

1. **Real Learning**: Agent discovers strategies on its own
2. **Full Transparency**: Every decision is logged with reasoning
3. **Transfer Ready**: Built for cross-domain knowledge sharing
4. **Production Quality**: Comprehensive tests, documentation, error handling
5. **Research Value**: All data saved for deep analysis

## Next Steps

1. **Run the demo** to see the game
2. **Train an agent** to see learning in action
3. **Analyze logs** to understand behavior
4. **Try transfer learning** to connect with other games
5. **Extend** with your own game environments!

---

**Questions?** See full docs or check the code - everything is documented!

**Want to add your own game?** Follow the transfer learning pattern - just map your concepts to abstractions!

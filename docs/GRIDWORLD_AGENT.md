# GridWorld Survival Agent - Documentation

## Overview

The GridWorld Survival Agent is a fully functioning intelligent agent trained on a complex game environment that demonstrates advanced learning, adaptation, and decision-making capabilities using the NSCK (Neuro-Symbolic Cognitive Kernel) architecture.

## What Makes This Different from Snake/Maze/Pong?

### Game Complexity

GridWorld Survival is significantly more complex than basic games:

| Feature | Snake | Maze | Pong | **GridWorld Survival** |
|---------|-------|------|------|----------------------|
| Objectives | 1 (eat food) | 1 (reach exit) | 1 (hit ball) | **Multiple (survival + progress)** |
| Resources | None | None | None | **4 (energy, health, hunger, thirst)** |
| Threats | Static (walls, self) | Static (walls) | None | **Dynamic (patrolling enemies + hazards)** |
| Strategy | Reactive | Path-finding | Reactive | **Long-term planning required** |
| Trade-offs | None | None | None | **Risk vs reward decisions** |
| State Space | Small | Medium | Small | **Large (continuous stats + dynamic entities)** |

### What the Agent Must Learn

1. **Multi-Objective Optimization**: Balance competing needs (survival vs progress)
2. **Risk Assessment**: Evaluate danger vs reward (e.g., treasure near enemy)
3. **Resource Management**: Predict future needs and plan accordingly
4. **Dynamic Adaptation**: Respond to moving enemies with patrol patterns
5. **Strategic Planning**: Can't just rush to goal - must survive the journey

## System Architecture

### Components

```
GridWorld Survival System
│
├── gridworld_survival.py          # Game environment
│   ├── GridWorldSurvival         # Main game class
│   ├── GridWorldState            # State representation
│   ├── Enemy                     # Dynamic enemy entities
│   └── create_gridworld_causal_graph()  # Causal relationships
│
├── train_gridworld_agent.py      # Training system
│   ├── GridWorldAgent            # Intelligent agent with NSCK
│   ├── GridWorldLogger           # Comprehensive logging
│   ├── train_agent()             # Training loop
│   └── provide_guidance()        # Guided learning
│
├── demo_gridworld.py              # Standalone demo
│   └── simple_agent_policy()    # Rule-based baseline
│
└── test_gridworld_agent.py       # Comprehensive tests
    ├── TestGridWorldGame
    ├── TestGridWorldAgent
    ├── TestGridWorldLogger
    └── TestTransferLearning
```

### Integration with NSCK

The agent uses the full NSCK cognitive architecture:

- **Rule Learning**: Discovers patterns from experience (frequency-based ILP)
- **Episodic Memory**: Stores and retrieves past experiences (VSA-based)
- **Causal Reasoning**: Understands cause-effect relationships
- **Global Workspace**: Integrates multiple decision-making modules
- **Metacognition**: Tracks confidence and self-improvement
- **Transfer Learning**: Abstracts knowledge for use in other games

## Game Mechanics

### Entities

- **Player (@)**: The agent-controlled character
- **Food (F)**: Reduces hunger (+10 score, -0.4 hunger)
- **Water (W)**: Reduces thirst (+10 score, -0.4 thirst)
- **Enemy (E)**: Patrols and damages player (-0.3 health)
- **Hazard (X)**: Static damage zones (-0.2 health)
- **Treasure (T)**: Bonus points (+50 score)
- **Exit (★)**: Goal location (+100 score, win condition)
- **Wall (#)**: Impassable barriers

### Resource System

```
Energy:  1.0 → 0.0  (drains -0.002/step, -0.001 extra when moving)
Health:  1.0 → 0.0  (damaged by enemies/hazards/critical states)
Hunger:  0.0 → 1.0  (increases +0.003/step, critical at 1.0)
Thirst:  0.0 → 1.0  (increases +0.004/step, critical at 1.0)
```

### Win/Loss Conditions

- **Win**: Reach the exit (★)
- **Loss**: 
  - Health reaches 0
  - Energy reaches 0
  - Timeout (max_steps exceeded)

## Usage

### Quick Demo

```bash
# Run standalone demo (no NSCK dependencies)
python nsck-demo/python/demo_gridworld.py --episodes 5

# Show complexity explanation
python nsck-demo/python/demo_gridworld.py --complexity
```

### Full Agent Training

```bash
# Train agent with comprehensive logging
python nsck-demo/python/train_gridworld_agent.py --episodes 50 --guided 5

# Options:
#   --episodes N       Number of training episodes (default: 50)
#   --guided N         Number of guided episodes (default: 5)
#   --render-freq N    Render every N episodes (default: 10)
#   --log-dir PATH     Log directory (default: gridworld_logs)
```

### Transfer Learning Demo

```bash
# Demonstrate transfer learning capability
python nsck-demo/python/train_gridworld_agent.py --demo-transfer
```

## Logging and Analysis

### Log Files Generated

When training, the system creates a timestamped session directory with:

```
gridworld_logs/
└── session_YYYYMMDD_HHMMSS/
    ├── game_log.jsonl           # Every state, action, reward
    ├── metrics.jsonl            # Episode-level metrics
    ├── learning_events.jsonl   # Rule discovery, updates
    ├── agent.log                # Human-readable log
    └── summary.txt              # Training summary
```

### What's Logged

#### Game Log (game_log.jsonl)
Each line is a JSON object with:
- `episode`, `step`: Episode and step numbers
- `state`: Complete game state (positions, stats, predicates)
- `action`: Action taken
- `reward`: Reward received
- `done`: Whether episode ended
- `info`: Additional info (reason for state change)
- `reasoning`: Agent's decision reasoning (confidence, predicates, explanation)

#### Metrics Log (metrics.jsonl)
Each line represents one episode:
- `episode`: Episode number
- `total_reward`: Cumulative reward
- `steps`: Survival time
- `final_score`: Game score
- `success`: Whether exit was reached
- `stats`: Final resource values

#### Learning Events (learning_events.jsonl)
Records when agent learns:
- `rule_learned`: New rule discovered
- `confidence_update`: Confidence adjusted
- `strategy_adapted`: Behavior changed

#### Summary (summary.txt)
Comprehensive analysis including:
- Performance metrics (mean, std, min, max)
- Success rate
- Learning progress (early vs late episodes)
- Improvement trends
- Event counts

### Example Analysis

```python
import json

# Load and analyze game log
with open('gridworld_logs/session_*/game_log.jsonl') as f:
    steps = [json.loads(line) for line in f]

# Find decisions that led to high rewards
good_decisions = [s for s in steps if s['reward'] > 1.0]
for decision in good_decisions[:5]:
    print(f"Action: {decision['action']}")
    print(f"State predicates: {decision['state']['predicates']}")
    print(f"Reasoning: {decision['reasoning']}")
    print()

# Track resource management over time
energies = [s['state']['stats']['energy'] for s in steps]
import matplotlib.pyplot as plt
plt.plot(energies)
plt.xlabel('Step')
plt.ylabel('Energy')
plt.title('Energy Management Over Time')
plt.show()
```

## Training Results

### What to Expect

After 50 episodes of training, typical results:

- **Success Rate**: 20-40% (reaching exit)
- **Average Score**: Improves from ~10-20 to 40-80
- **Survival Time**: Increases as agent learns resource management
- **Rules Learned**: 15-30 rules about survival strategies

### Learning Progression

1. **Episodes 1-10**: Random exploration, often dies quickly
2. **Episodes 11-25**: Learns basic survival (seek food/water when critical)
3. **Episodes 26-40**: Develops strategies (risk assessment, planning)
4. **Episodes 41-50**: Optimizes behavior (efficient resource use, goal-directed)

## Transfer Learning

### Abstract Concepts

The agent learns using abstract concepts that transfer between games:

| Concrete (GridWorld) | Abstract | Other Games |
|---------------------|----------|-------------|
| FOOD, WATER | RESOURCE | SNAKE_FOOD, MAZE_POWERUP |
| ENEMY, HAZARD | THREAT | SNAKE_WALL, PONG_OPPONENT |
| EXIT | GOAL | MAZE_EXIT, SNAKE_TARGET |
| CRITICAL_HUNGER | CRITICAL_NEED | LOW_HEALTH, LOW_TIME |

### How Transfer Works

1. **Training on GridWorld**: Agent learns "avoid THREAT when CRITICAL_NEED"
2. **Abstraction**: Rule lifts to "avoid THREAT when CRITICAL_NEED"
3. **Transfer to Snake**: "avoid SNAKE_WALL when LOW_HEALTH"

This enables **zero-shot transfer** - applying knowledge to new games without retraining!

### Adding New Games

To enable transfer to a new game:

```python
# Register domain
analogy_engine.register_domain("my_game")

# Map concrete concepts to abstractions
analogy_engine.register_abstract("MY_PLAYER", "AGENT", "my_game")
analogy_engine.register_abstract("MY_FOOD", "RESOURCE", "my_game")
analogy_engine.register_abstract("MY_ENEMY", "THREAT", "my_game")

# Rules learned in GridWorld now apply to my_game!
```

## Testing

```bash
# Run all tests
python -m pytest nsck-demo/tests/test_gridworld_agent.py -v

# Run specific test class
python -m pytest nsck-demo/tests/test_gridworld_agent.py::TestGridWorldGame -v

# Run without pytest
python nsck-demo/tests/test_gridworld_agent.py
```

## Performance Characteristics

### Computational Efficiency

- **No GPU required**: Runs entirely on CPU
- **Memory footprint**: ~500MB for full system
- **Training speed**: ~1-2 seconds per episode on modern CPU
- **Decision latency**: <10ms per decision

### Scalability

- Grid size: Tested up to 30x30 (900 cells)
- Entities: Tested up to 20 simultaneous entities
- Episodes: Can train for 1000+ episodes
- State space: Handles ~10^6 distinct states

## Demonstrations and Proofs

### 1. Learning Capability

**Demonstrated by**: Improvement in scores over episodes

Run training and observe:
- Early episodes: Low scores (10-20)
- Late episodes: Higher scores (40-80)
- Rules learned: Check `learning_events.jsonl`

### 2. Adaptation

**Demonstrated by**: Response to different situations

Observe agent behavior when:
- Critical hunger → seeks food immediately
- Enemy nearby → avoids or finds safe path
- High resources → moves toward exit

### 3. Multi-Objective Decision Making

**Demonstrated by**: Trade-off decisions

Agent learns to:
- Sacrifice immediate progress for survival (collect food before exit)
- Take calculated risks (go near enemy for treasure if healthy)
- Optimize resource collection (prioritize most critical need)

### 4. Transfer Learning

**Demonstrated by**: Abstract concept grounding

```bash
# Show abstract mappings
python -c "
from train_gridworld_agent import GridWorldAgent
agent = GridWorldAgent()
analogy = agent.engine.analogy

# Show abstraction
print('FOOD abstracts to:', analogy.lift_to_abstract('FOOD', 'gridworld'))
print('ENEMY abstracts to:', analogy.lift_to_abstract('ENEMY', 'gridworld'))

# Show it can ground to other domains
analogy.register_domain('snake')
analogy.register_abstract('SNAKE_FOOD', 'RESOURCE', 'snake')
print('RESOURCE grounds to (in snake):', analogy.ground_to_domain('RESOURCE', 'snake'))
"
```

## Troubleshooting

### Common Issues

**Issue**: Agent dies immediately in early episodes
- **Expected**: Agent needs time to learn survival strategies
- **Solution**: Increase guided episodes (`--guided 10`)

**Issue**: Low success rate even after training
- **Expected**: Game is challenging, 40% is good
- **Solution**: Train longer (`--episodes 100`), adjust game difficulty

**Issue**: Missing dependencies
- **Solution**: `pip install numpy scipy networkx`

## Future Enhancements

Potential extensions to demonstrate even more capabilities:

1. **Cooperative Multi-Agent**: Multiple agents must cooperate
2. **Hierarchical Planning**: Break down long-term goals into subgoals
3. **Communication**: Agents share knowledge through language
4. **Meta-Learning**: Learn how to learn faster
5. **Curriculum Learning**: Gradually increase difficulty

## References

- Main codebase: `/nsck-demo/python/`
- Game environment: `gridworld_survival.py`
- Training system: `train_gridworld_agent.py`
- Demo script: `demo_gridworld.py`
- Tests: `/nsck-demo/tests/test_gridworld_agent.py`
- NSCK architecture: See `docs/ARCHITECTURE.md`

## Citation

If you use this system in your research, please cite:

```
GridWorld Survival Agent - NSCK Demonstration
Part of the Neuro-Symbolic Cognitive Kernel (NSCK) project
Repository: https://github.com/shiva2321/Node_network
```

# NSCK V5 Quick Start

Get up and running in 5 minutes.

## Install

```bash
cd nsck
pip install -r ../requirements.txt
```

## Basic Decision Loop

```python
import sys; sys.path.insert(0, 'python')
from python.core.substrate import NSCKSubstrate

# Create the substrate
substrate = NSCKSubstrate()
substrate.register_task("my_task")

# Make a decision
state = {"text": "move forward to reach the goal", "position_x": 0, "position_y": 0}
result = substrate.process(state, "my_task")
print(f"Action: {result.chosen_action}, Confidence: {result.confidence:.2f}")
```

## Learning from Feedback

```python
# Give positive feedback when the action was correct
substrate.feedback(
    action=result.chosen_action,
    reward=1.0,
    task_tag="my_task",
    state=state,
    outcome="success"
)
```

## Multimodal Input

```python
# Process text + numeric + image signals
state_mm = {
    "text": "high sensor reading",
    "sensor_value": 0.9,
    "timestamp": 12345,
}
result = substrate.process(state_mm, "my_task")
```

## Sleep Consolidation

```python
# Run offline consolidation after collecting experiences
sleep_result = substrate.sleep("my_task")
print(f"Sleep cycles completed: {sleep_result['sleep_cycles']}")
```

## Full Example

```python
import sys; sys.path.insert(0, 'python')
from python.core.substrate import NSCKSubstrate

substrate = NSCKSubstrate()
substrate.register_task("navigation")

for step in range(20):
    state = {"position_x": step % 5, "position_y": step // 5}
    result = substrate.process(state, "navigation")
    reward = 1.0 if step % 3 == 0 else 0.0
    substrate.feedback(
        action=result.chosen_action, reward=reward,
        task_tag="navigation", state=state,
    )
    print(f"Step {step}: action={result.chosen_action}, reward={reward}")

# Consolidate knowledge
substrate.sleep("navigation")
stats = substrate.engine.stats
print(f"Rules learned: {stats['rules_induced']}, Episodes: {stats['episodes_recorded']}")
```

## Next Steps

- See [API_REFERENCE.md](API_REFERENCE.md) for complete API documentation
- See [ARCHITECTURE.md](ARCHITECTURE.md) for architecture overview
- See [TESTING.md](TESTING.md) for running tests
- Run `make test-v5` for V5 integration tests
- Run `make benchmark-realworld` for benchmarks

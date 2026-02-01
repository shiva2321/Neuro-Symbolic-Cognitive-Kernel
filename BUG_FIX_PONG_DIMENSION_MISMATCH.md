# Bug Fix: Pong Dimension Mismatch Error

## Date
February 1, 2026

## Issue
When the Pong game was started alongside Snake and Maze, the system crashed with a RuntimeError:

```
RuntimeError: The size of tensor a (4) must match the size of tensor b (2) at non-singleton dimension 1
```

Error location: `python_server.py` line 1239

## Root Cause
The VSA (Vector Symbolic Architecture) prior system was trying to apply action space priors across different games with different action dimensions:

- **Snake/Maze**: 4 actions (UP, DOWN, LEFT, RIGHT)
- **Pong**: 2 actions (UP, DOWN)

When computing `biased_probs = snn_probs * (1.0 + VSA_STRENGTH * prior_tensor)`, the system attempted to multiply tensors of incompatible shapes:
- `snn_probs`: shape [1, 2] for Pong
- `prior_tensor`: shape [1, 4] from Snake/Maze context

## Solution
Added a dimension check before applying VSA bias in `python_server.py` around line 1239:

```python
# Apply Bias - but only if dimensions match
if prior_tensor.shape[1] == snn_probs.shape[1]:
    biased_probs = snn_probs * (1.0 + VSA_STRENGTH * prior_tensor)
    probs = biased_probs / biased_probs.sum(dim=1, keepdim=True)
    prior_active = True
else:
    # Dimension mismatch - skip VSA rescue for this game
    pass
```

## Impact
- **Fixed**: Pong can now be launched without crashing the system
- **Graceful Degradation**: When VSA priors don't match the action space, the system falls back to using pure SNN predictions
- **No Breaking Changes**: Snake and Maze continue to benefit from VSA priors as before

## Testing Recommendation
1. Start all three games simultaneously (Snake, Maze, Pong)
2. Verify no dimension mismatch errors occur
3. Confirm each game receives appropriate actions from its neural network segment
4. Monitor that VSA rescue still works for Snake and Maze

## Future Considerations
Consider implementing game-specific VSA priors that automatically adapt to each game's action space, or create a unified action representation that works across all games.

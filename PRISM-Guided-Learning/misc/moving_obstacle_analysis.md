# Moving Obstacle Implementation Analysis

Analysis of `counterfactural learning withPMC.py`

## Summary

The moving obstacle implementation has a fundamental design mismatch between the RL environment and the PRISM verification model.

---

## RL Agent's Handling of Moving Obstacles

### Deterministic Cyclic Movement

The obstacle moves one step forward in its 15-position path each time the agent takes an action:

```python
def _update_moving_obstacle(self):
    self.current_moving_index = (self.current_moving_index + 1) % len(self.moving_obstacle_positions)
    self.current_moving_obstacle = self.moving_obstacle_positions[self.current_moving_index]
```

Path: `(7,4) -> (7,3) -> (7,2) -> (6,2) -> (5,2) -> (6,2) -> (7,2) -> (7,3) -> (7,4) -> (7,5) -> (7,6) -> (7,7) -> (7,8) -> (7,9) -> (8,9) -> [repeat]`

### Collision Detection

The agent checks collision with the **current** obstacle position:

```python
if ((new_x, new_y) in self.static_obstacles or
    (new_x, new_y) == self.current_moving_obstacle):
    new_x, new_y = x, y  # Agent stays in place
    reward = -10.0       # Collision penalty
```

### State Representation

The state is `(x, y, g1, g2, g3)` - the agent does **not** observe where the obstacle is.

---

## PRISM Model's Handling of Moving Obstacles

### No Separate Agent/Module

The moving obstacle is **not** encoded as a separate PRISM module with its own state. The model has only one module (`gridworld`) with variables:
- `x`, `y` - agent position
- `g1`, `g2`, `g3` - goal reached flags

There is no variable tracking the obstacle's position.

### Probabilistic Abstraction

When generating transitions, if the agent moves to ANY position on the obstacle's path:
- **10% probability**: Agent stays in place (blocked)
- **90% probability**: Agent moves to the next position in the obstacle's cyclic path

```python
if (new_x, new_y) == current_obs_pos:
    transition_prob = 0.9
    stay_prob = 0.1
    # Split into two probabilistic outcomes
```

### Label Definition

The `at_obstacle` label includes ALL positions in the path, not a specific current position:

```python
moving_obstacle_label = " | ".join([f"(x={x} & y={y})"
                                  for x, y in self.moving_obstacle_positions])
```

---

## Comparison Table

| Aspect | RL Environment | PRISM Model |
|--------|---------------|-------------|
| Obstacle tracking | Exact position via `current_moving_index` | No position tracking |
| Movement | Deterministic, advances each step | Probabilistic abstraction (90%/10%) |
| Agent state | `(x, y, g1, g2, g3)` | `(x, y, g1, g2, g3)` |
| Collision | Checks current position exactly | Checks all path positions probabilistically |

---

## Problems

1. **Agent is "blind" to obstacle**: State doesn't include obstacle position, so agent cannot learn timing-aware avoidance strategies.

2. **Q-table limitation**: Same agent position maps to same Q-values regardless of where the obstacle is.

3. **RL/PRISM mismatch**: The RL environment and PRISM model use fundamentally different obstacle dynamics.

4. **Verification accuracy**: PRISM verification doesn't reflect actual obstacle behavior - uses an over-approximation.

---

## Potential Fix

To properly handle moving obstacles, the obstacle index should be added to the state:

```python
# State becomes: (x, y, obs_idx, g1, g2, g3)
state = (x, y, self.current_moving_index, g1, g2, g3)
```

And the PRISM model should use a separate synchronized module:

```prism
module obstacle
  obs_idx : [0..14] init 0;
  [step] true -> (obs_idx'=mod(obs_idx+1, 15));
endmodule

module agent
  x : [0..9] init 0;
  y : [0..9] init 0;
  g1 : bool init false;
  // ... transitions synchronized with [step] action
endmodule
```

This would allow:
- Agent to learn obstacle-aware policies
- Accurate PRISM verification of synchronized behavior
- Proper temporal reasoning about obstacle positions

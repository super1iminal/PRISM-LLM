# Plan: Moving Obstacles in PRISM + Per-Segment Feedback

## Summary
Replace the broken moving obstacle implementation with a synchronized PRISM obstacle module. Add per-segment collision queries and per-segment LLM feedback.

## Design Decisions
- **Data**: Add `moving` column to CSV; backward-compatible (default `[]` if missing)
- **Collision model**: Agent passes through freely — NO blocking. Collision is tracked purely via PRISM labels (agent at same cell as obstacle at same time step). This keeps transitions simple.
- **No `at_obstacle`/`avoid_obstacle`**: Static obstacle avoidance is trivially 1.0 (agent bounces back, can never occupy a static obstacle cell). The old global `avoid_obstacle` property and `at_obstacle` label are **removed entirely**. Replaced by per-segment `at_moving_obs` collision properties only.
- **Feedback**: Per-segment probabilities PLUS global metrics (complete_sequence)
- **Path expansion**: Non-looped paths become back-and-forth (e.g. `[A,B,C]` → `[A,B,C,B]`)
- **Replace, not extend**: The old probabilistic obstacle abstraction (90/10 split, agent displaced to next obstacle position) is **completely removed**. No remnants of the old approach should remain.

---

## Step 0: GridWorld Path Expansion

**File: `src/environment/GridWorld.py`**

Add `_expand_path()` to handle back-and-forth:
- If `path[-1] == path[0]`: already looped, strip duplicate endpoint → `path[:-1]`
- Else: back-and-forth → `path + path[-2:0:-1]` (reverse middle, skip endpoints)
- Store result in `self.moving_obstacle_positions` (overwrite raw input)
- Store `self.moving_obstacle_path_raw` for reference
- Store `self.num_obs_steps = len(self.moving_obstacle_positions)` (0 if empty)

## Step 1: DataLoader CSV Parsing

**File: `src/utils/DataLoader.py`**

- Parse optional `moving` column with `ast.literal_eval`, default to `[]`
- Pass to `GridWorld(size, goals, obstacles, moving)`

## Step 2: Synchronized PRISM Obstacle Module

**File: `src/verification/PrismModelGenerator.py`**

This step **replaces** the entire current moving obstacle handling. The old code (lines 63-86 in `_generate_transitions()` — the 90/10 probabilistic abstraction that displaced the agent to the next obstacle position) is **deleted entirely**.

### 2a. `generate_prism_model()` — Add obstacle module

When `moving_obstacle_positions` is non-empty, generate a second synchronized module:

```prism
dtmc
const int N = 5;

module gridworld
  x : [0..4] init 0;
  y : [0..4] init 0;
  g1 : bool init false;
  g2 : bool init false;

  [step] (x=0 & y=0 & g1=false & g2=false) -> 0.7:(...) + 0.15:(...) + 0.15:(...);
  // ... all transitions use [step] label for synchronization
endmodule

module obstacle
  obs_idx : [0..3] init 0;
  [step] (obs_idx < 3) -> (obs_idx'=obs_idx+1);
  [step] (obs_idx = 3) -> (obs_idx'=0);
endmodule
```

- Use `[step]` action label on ALL transitions (both modules) for synchronization
- When no moving obstacles: use `[]` label as before (backward compatible)
- Use two guarded commands for wrapping (PRISM doesn't support `mod()` in updates)

### 2b. `_generate_transitions()` — Remove old obstacle handling, simplify

**Delete entirely**: The old moving obstacle block (lines 63-86) that checked `(new_x, new_y) == current_obs_pos` and did the 90/10 split.

Since the agent passes through freely (no blocking on collision), the transition logic becomes simpler:
- Agent always moves to its intended cell (or bounces off walls/static obstacles as before)
- Moving obstacle cells are treated as **normal passable cells** in the transition logic
- The `at_moving_obs` label tracks collision purely at the label level

The only change to transitions is the guard prefix:
- With moving obstacles: `[step] (x=... & y=... & g1=... & g2=...) ->`
- Without moving obstacles: `[] (x=... & y=... & g1=... & g2=...) ->` (unchanged)

No `obs_idx` in agent guards. No per-obs_idx transition duplication. This keeps the model compact.

### 2c. Labels

**Remove entirely**: The old `at_obstacle` label (lumped all moving+static obstacle positions). Static obstacle avoidance is trivially 1.0 since transitions bounce the agent back — the agent can never occupy a static obstacle cell. So this label was dead weight.

**Add new label** (only when moving obstacles exist):

```prism
// Moving obstacle collision (position + time synchronized)
// Agent is at the same cell as the obstacle at the same time step
label "at_moving_obs" =
  (obs_idx=0 & x=0 & y=2) |
  (obs_idx=1 & x=1 & y=2) |
  ...;
```

When no moving obstacles: no obstacle labels are emitted at all (no `at_obstacle`, no `at_moving_obs`).

**Segment labels** (for per-segment queries):
```prism
label "in_seg1" = !g1;
label "in_seg2" = g1 & !g2;
label "in_seg3" = g2 & !g3;
```

## Step 3: Per-Segment PRISM Properties

**File: `src/verification/SimplifiedVerifier.py`**

**Remove**: The old `avoid_obstacle` requirement (`P=? [ G<=30 !"at_obstacle" ];` with weight 0.25). Static obstacle avoidance is trivially 1.0 and the old `at_obstacle` label is gone.

**Add** per-segment moving obstacle collision requirements at the END of `_build_requirements()`:

```python
# Per-segment moving obstacle avoidance (replaces old avoid_obstacle)
if self.gridWorld.moving_obstacle_positions:
    seg_weight = 0.25 / len(goal_list)  # distribute old 0.25 safety weight across segments
    for idx, goal_num in enumerate(goal_list):
        requirements.append(Requirement(
            name=f"Avoid moving obs (seg {goal_num})",
            property=f'P=? [ G ("in_seg{goal_num}" => !"at_moving_obs") ];',
            weight=seg_weight,
            category="segment_safety",
            csv_header=f"Avoid_Moving_Seg{goal_num}"
        ))
```

When no moving obstacles: no safety requirements at all (no `avoid_obstacle`, no per-segment). The weight distribution adjusts: reachability and sequence get the full 1.0 budget. Update weight calculations accordingly.

Segment labels are generated in `PrismModelGenerator.generate_prism_model()` (Step 2c).

## Step 4: Parse New Probabilities

**Files: ALL 5 planners** (identical change in each):
- `src/planners/VanillaLLMPlanner.py`
- `src/planners/FeedbackLLMPlanner.py`
- `src/planners/FeedbackMinusLLMPlanner.py`
- `src/planners/VanillaPlusLLMPlanner.py`
- `src/planners/RLCounterfactual.py`

In `_update_prism_probabilities()`:

1. **Remove** the `avoid_obstacle` parsing (the old `self.prism_probs['avoid_obstacle'] = probabilities[idx]`)
2. **Add** per-segment parsing at the end:

```python
# Per-segment moving obstacle avoidance (replaces old avoid_obstacle)
if self.env.moving_obstacle_positions:
    for goal_num in goal_nums:
        if idx < len(probabilities):
            self.prism_probs[f'avoid_moving_seg{goal_num}'] = probabilities[idx]
            idx += 1
```

**Evaluator.py** needs NO changes — the dynamic `**{f"prob_{k}": v for k, v in prism_probs.items()}` on line 241 automatically picks up new keys as Parquet columns. Old `prob_avoid_obstacle` column simply won't appear anymore.

**RLCounterfactual._init_prism_probs()**: Remove `avoid_obstacle` init, add per-segment keys initialized to 0.0 (when moving obstacles present).

## Step 5: Per-Segment LLM Feedback

### 5a. `src/utils/LLMPrompting.py`

Add `extract_segment_probs()`:
```python
def extract_segment_probs(prism_probs, goal_num, goal_nums):
    """Extract probabilities relevant to a specific goal segment."""
    segment = {}
    # Goal reachability
    if f'goal{goal_num}' in prism_probs:
        segment[f'goal{goal_num}'] = prism_probs[f'goal{goal_num}']
    # Sequence ordering for this segment
    idx = goal_nums.index(goal_num)
    if idx > 0:
        prev = goal_nums[idx - 1]
        key = f'seq_{prev}_before_{goal_num}'
        if key in prism_probs:
            segment[key] = prism_probs[key]
    # Per-segment moving obstacle avoidance
    seg_key = f'avoid_moving_seg{goal_num}'
    if seg_key in prism_probs:
        segment[seg_key] = prism_probs[seg_key]
    return segment
```

Update `identify_problems()`:
1. **Remove** the `elif key == 'avoid_obstacle'` case
2. **Add** handling for `avoid_moving_seg*` keys:
```python
elif key.startswith('avoid_moving_seg'):
    seg_num = key.split('seg')[1]
    problems.append(f"- Moving obstacle collision risk (segment to goal {seg_num}): {prob:.4f} < 1.0 - Path to goal {seg_num} crosses moving obstacle trajectory")
```

### 5b. `src/planners/FeedbackLLMPlanner.py` + `FeedbackMinusLLMPlanner.py`

In `_get_feedback_prompt()`, build per-segment + global feedback:

```python
goal_nums = sorted(self.env.goals.keys())
goal_num = goal_nums[goal_idx]

# Per-segment probs for this goal
segment_probs = extract_segment_probs(self.prism_probs, goal_num, goal_nums)

# Also include global metrics (avoid_obstacle is gone, only complete_sequence remains)
if 'complete_sequence' in self.prism_probs:
    segment_probs['complete_sequence'] = self.prism_probs['complete_sequence']

prob_summary = format_probability_summary(segment_probs)
problems = identify_problems(segment_probs, self.target_threshold)
```

---

## Files Modified (in implementation order)

| # | File | Change |
|---|------|--------|
| 1 | `src/environment/GridWorld.py` | Add `_expand_path()`, store expanded cycle |
| 2 | `src/utils/DataLoader.py` | Parse optional `moving` CSV column |
| 3 | `src/verification/PrismModelGenerator.py` | **Replace** old moving obstacle code: add synchronized obstacle module, `[step]` labels, remove 90/10 abstraction, add `at_moving_obs` label, add segment labels |
| 4 | `src/verification/SimplifiedVerifier.py` | Remove `avoid_obstacle`, add per-segment collision requirements |
| 5 | `src/planners/VanillaLLMPlanner.py` | Update `_update_prism_probabilities` |
| 6 | `src/planners/FeedbackLLMPlanner.py` | Update `_update_prism_probabilities` + per-segment feedback |
| 7 | `src/planners/FeedbackMinusLLMPlanner.py` | Same as above |
| 8 | `src/planners/VanillaPlusLLMPlanner.py` | Update `_update_prism_probabilities` |
| 9 | `src/planners/RLCounterfactual.py` | Update `_update_prism_probabilities` + `_init_prism_probs` |
| 10 | `src/utils/LLMPrompting.py` | Add `extract_segment_probs()`, update `identify_problems()` |
| 11 | `data/*.csv` | Add `moving` column (empty `[]` for existing data) |

## Verification Plan

1. **Unit check**: Create a small 3x3 grid with a 2-position moving obstacle. Generate the PRISM model string and inspect that:
   - The obstacle module exists with correct `obs_idx` range
   - `[step]` labels are on all transitions
   - `at_moving_obs` label correctly pairs obs_idx with positions
   - Segment labels are present
   - No remnants of old 90/10 obstacle handling

2. **PRISM execution**: Run PRISM on the generated model to verify it parses and produces results for all properties (existing + new per-segment ones)

3. **Backward compatibility**: Run with `moving=[]` and verify output is identical to current behavior (no obstacle module, `[]` labels, same probabilities)

4. **End-to-end**: Run `Evaluator.py` on a small dataset with moving obstacles and verify:
   - Parquet output has new `prob_avoid_moving_seg*` columns
   - Feedback prompts show per-segment probabilities
   - No errors in any planner

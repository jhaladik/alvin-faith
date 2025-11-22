# Critical Reward Mismatch Between Training and Visual Game

## Problem Discovered

The visual game (`expanded_faith_visual_games.py`) uses **completely different rewards** than the training script (`train_snake_improved.py`). This causes the agent to behave incorrectly!

## Comparison Table

| Component | Training (train_snake_improved.py) | Visual Game (expanded_faith_visual_games.py) | Match? |
|-----------|-----------------------------------|---------------------------------------------|--------|
| **Pellet Collection** | 50 base + combo × 10 | 10 base + combo × 2 (×risk multiplier) | ❌ NO |
| **Survival per step** | +0.1 | +0.05 | ❌ NO |
| **Death penalty** | -100 | -(50 + steps × 0.1) | ❌ NO |
| **Danger zone** | -1.0 when <1.5 tiles from wall | NOT IMPLEMENTED | ❌ NO |
| **Approach food** | +0.5 toward, -0.2 away | +0.5 closer, -0.1 farther | ⚠️ SIMILAR |

## Critical Issues

### 1. Pellet Reward Mismatch (5x difference!)

**Training:**
```python
pellet_reward = 50.0 + (self.combo_count * 10.0)
# First pellet: 50
# Second: 60
# Third: 70
```

**Visual:**
```python
base_pellet = 10.0
combo_bonus = self.combo_count * 2.0
total = (base_pellet + combo_bonus) * risk_mult
# First pellet: 10 × 1 = 10
# Second: 12 × 1 = 12
# Third: 14 × 1 = 14
```

**Effect:** Agent trained to value pellets at 50-70, but visual game gives 10-14. Agent's Q-values will be completely wrong!

### 2. Danger Zone Missing in Visual

**Training:**
```python
# Agent learned to avoid being near walls
if min_wall_dist < 1.5:
    danger_penalty = -1.0 * (1.0 - min_wall_dist / 1.5)
```

**Visual:**
```python
# NO danger zone penalty!
# Agent doesn't get feedback for being near walls
```

**Effect:** Agent's learned behavior around walls won't match the visual environment.

### 3. Death Penalty Mismatch (2x different)

**Training:**
```python
if died:
    total -= 100.0  # Fixed -100
```

**Visual:**
```python
if died:
    breakdown['death_penalty'] = -(50.0 + self.steps_alive * 0.1)
    # After 100 steps: -(50 + 10) = -60
    # After 200 steps: -(50 + 20) = -70
```

**Effect:** Agent underestimates death penalty in visual game.

### 4. Survival Reward Mismatch (2x different)

**Training:** +0.1 per step
**Visual:** +0.05 per step

**Effect:** Agent overvalues survival in visual game.

## Why This Causes Problems

### Q-Value Mismatch

Agent's Q-network learned during training:
```
Q(state, action) ≈ expected_sum_of_training_rewards
```

But in visual game, actual rewards are different:
```
actual_visual_rewards ≠ training_rewards
```

Result: **Agent's decisions are suboptimal because Q-values don't match actual rewards!**

### Example Scenario

**Situation:** Snake near pellet, 2 tiles from wall

**Agent's Q-network thinks (from training):**
- Collect pellet: +50 reward
- Stay near wall: -0.5 danger penalty
- Net expected: +49.5

**Visual game actually gives:**
- Collect pellet: +10 reward (5x less!)
- Stay near wall: 0 danger penalty (not implemented!)
- Net actual: +10

Agent will be "confused" - it expected +49.5 but got +10!

## Solutions

### Solution 1: Use Training Rewards in Visual (Recommended)

Make visual game match training exactly:

```python
# In expanded_faith_visual_games.py, replace ContinuousMotivationRewardSystem

class SnakeTrainingMatchedRewardSystem:
    """Matches train_snake_improved.py exactly"""

    def __init__(self):
        self.combo_count = 0
        self.steps_alive = 0

    def reset(self):
        self.combo_count = 0
        self.steps_alive = 0

    def calculate(self, base_reward, died, min_wall_dist,
                  nearest_food_dist=None, prev_nearest_food_dist=None):
        total = 0.0

        # Match training: Pellet 50 + combo * 10
        if base_reward >= 10:
            pellet_reward = 50.0 + (self.combo_count * 10.0)
            total += pellet_reward
            self.combo_count += 1

        # Match training: Survival +0.1
        self.steps_alive += 1
        total += 0.1

        # Match training: Death -100
        if died:
            total -= 100.0

        # Match training: Danger zone
        if min_wall_dist < 1.5:
            danger_penalty = -1.0 * (1.0 - min_wall_dist / 1.5)
            total += danger_penalty

        # Match training: Approach food
        if nearest_food_dist is not None and prev_nearest_food_dist is not None:
            if nearest_food_dist < prev_nearest_food_dist:
                total += 0.5
            elif nearest_food_dist > prev_nearest_food_dist:
                total -= 0.2

        return total
```

### Solution 2: Retrain with Visual Rewards

Retrain agent using visual game's reward structure. But this loses the improvements we made!

### Solution 3: Use Base Game Rewards Only

Don't use ANY enhanced rewards in visual - just show base game rewards:
- Pellet: +20 (from SnakeGame)
- Movement: +0.1 (from SnakeGame)
- Death: -50 (from SnakeGame)

But this means agent won't see the shaping that helped it learn.

## Recommendation

**Use Solution 1:** Update visual game to match training rewards exactly.

This ensures:
✓ Agent's Q-values are meaningful
✓ Agent behaves as trained
✓ Visual demo accurately shows trained behavior

The current mismatch is why the agent might look like it's making "bad" decisions in the visual game - it's actually following Q-values learned from different reward signals!

## Base Game Rewards (SnakeGame)

For reference, the base game rewards that BOTH should be enhancing:

```python
# From planning_test_games.py
movement: +0.1
food_collected: +20.0
wall_collision: -50.0 (then respawn)
all_food_collected: +200.0 (victory)
```

Both training and visual add enhancements on top of these base rewards, but they enhance them differently!

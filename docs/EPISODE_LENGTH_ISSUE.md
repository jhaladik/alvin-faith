# Episode Length Issue Analysis

## Observation

```
Episode 50:  Length 35.0,  Score 2.10   [SMALL 10x10]
Episode 100: Length 39.5,  Score 2.50   [SMALL 10x10]
Episode 150: Length 43.1,  Score 3.23   [MEDIUM 15x15]
Episode 200: Length 96.7,  Score 4.44   [MEDIUM 15x15]  ← BIG JUMP
Episode 250: Length 134.6, Score 5.89  [MEDIUM 15x15]  ← STILL GROWING
```

**Episode length is increasing faster than score!**

## Root Cause Analysis

### Game Termination Conditions

From `planning_test_games.py:188-196`:
```python
# Episode ends when:
1. All pellets collected (victory) → done = True
2. All lives lost (death) → done = True
3. steps >= max_steps (1000) → done = True
```

### What's Actually Happening

**Medium grid (15x15):** 7 pellets total
**Agent collecting:** 5-7 pellets (average 5.89 at episode 250)

**Problem:** Agent is NOT collecting all pellets!

When agent collects 5-7 out of 7 pellets:
- ✗ NOT victory (pellets remain)
- ✓ NOT dying (0 wall collisions)
- → Episode continues until **max_steps (1000)**!

### Why Episode Length Grows

1. **Agent learned to survive** (0 wall collisions) ✓
2. **Agent collects SOME food** (score 5-7) ✓
3. **But agent is WANDERING** instead of collecting remaining food ✗
4. **Episodes run until timeout** (max_steps) instead of victory ✗

**Average 134 steps to collect 5.89 pellets = ~23 steps per pellet**
That's VERY inefficient! Optimal would be ~10 steps per pellet.

### The Danger-Zone Problem

The danger-zone penalty might be **TOO STRONG**:

```python
# NEW: Danger zone penalty (continuous feedback)
if min_wall_dist < 2.0:  # Within 2 tiles of wall
    danger_penalty = -2.0 * (1.0 - min_wall_dist / 2.0)
```

**Effect:**
- Agent avoids walls aggressively ✓
- But also avoids **food near walls** ✗
- Wanders in "safe zones" in middle of grid
- Inefficient paths, long episodes

### Mathematical Analysis

**Expected episode length:**
- 7 pellets × 10 steps each = ~70 steps (efficient)
- Current: 134 steps (inefficient, 2x longer!)

**Why wandering?**
- Danger penalty: -2.0 when near walls
- Food reward: +50 (one-time)
- Movement penalty from danger: -2.0 × many steps
- **Agent learns: "Stay away from walls" > "Collect food near walls"**

### Evidence from Rewards

```
Episode 50:  Avg Reward -185.26
Episode 250: Avg Reward +156.97
```

Rewards ARE improving! But slowly, because:
- Agent collects food → +50 per pellet = +350 for 7 pellets
- But accumulates danger penalties: -2.0 × 134 steps near walls = -268
- Net: +82 after movement rewards
- **Danger penalties are eating into food rewards!**

## Solutions

### Option 1: Reduce Danger-Zone Penalty (Quick Fix)

```python
# Current (TOO STRONG):
if min_wall_dist < 2.0:
    danger_penalty = -2.0 * (1.0 - min_wall_dist / 2.0)

# Proposed (GENTLER):
if min_wall_dist < 1.5:  # Only penalize when VERY close
    danger_penalty = -1.0 * (1.0 - min_wall_dist / 1.5)  # Weaker penalty
```

Effect: Agent will tolerate being closer to walls to reach food

### Option 2: Add Efficiency Reward

```python
# Reward moving TOWARD food
nearest_food_dist = calculate_distance_to_nearest_food()
if nearest_food_dist < prev_nearest_food_dist:
    reward += 0.5  # Bonus for approaching food!
```

Effect: Encourages efficient pathfinding

### Option 3: Penalize Time Wastage

```python
# Add step penalty to encourage efficiency
reward -= 0.05  # Small penalty per step

# Or penalize when NOT collecting food
if steps_since_last_food > 50:
    reward -= 1.0  # "Hurry up!"
```

Effect: Agent learns to collect food quickly

### Option 4: Increase Max Steps in Early Training

```python
# Current:
game.max_steps = 1000  # Same for all episodes

# Proposed:
if episode < 200:
    game.max_steps = 100  # Force early termination
else:
    game.max_steps = 1000
```

Effect: Agent can't wander indefinitely early on

### Option 5: Multi-Objective Reward

```python
# Balance survival AND efficiency
efficiency_bonus = (score / steps) * 100  # Reward score-per-step
danger_penalty = min_wall_dist_penalty * 0.5  # Reduce weight

total_reward = food_reward + survival_reward + efficiency_bonus + danger_penalty
```

## Recommended Fix

**Reduce danger-zone penalty by 50% AND add approach-food reward:**

```python
# 1. Gentler danger zone
if min_wall_dist < 1.5:  # Was 2.0
    danger_penalty = -1.0 * (1.0 - min_wall_dist / 1.5)  # Was -2.0

# 2. Reward approaching food
nearest_food = min([distance_to(food) for food in food_positions])
if nearest_food < prev_nearest_food:
    reward += 0.5  # Approach bonus
```

## Expected Improvement

With fixes:
- Episode 250: Length ~70 steps (vs 134)
- Episode 250: Score ~7 pellets (vs 5.89)
- Episode 250: More victories, fewer timeouts
- Better balance: survival AND efficiency

## Current Status

**Good news:**
✓ Agent learned wall avoidance (0 collisions)
✓ Agent collecting food (score improving)
✓ Training is stable (rewards increasing)

**Bad news:**
✗ Agent wandering inefficiently
✗ Episodes too long (134 vs ~70 optimal)
✗ Danger penalties too strong
✗ Not learning efficient pathfinding

**The agent is overfitting to "don't die" and underfitting to "collect food efficiently"!**

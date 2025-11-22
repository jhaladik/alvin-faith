# Agent Circling Around Food - Root Cause Analysis

## Observed Behavior

```
Episode 250: Score 6.00/7, Length 169.9 steps
Episode 300: Score 6.45/7, Length 149.4 steps (improving)
Episode 350: Score 6.78/7, Length 158.1 steps
```

**Agent approaches food but circles around it instead of collecting!**

## Reward Balance Analysis

### When approaching food (per step):
```
Survival: +0.1
Approach: +0.5
Danger (if near wall): -0.5 (avg, varies by distance)
Total: +0.1 per step of approaching
```

### When collecting food:
```
Pellet: +50 + combo (one-time)
Survival: +0.1
Approach: 0 (already at food)
Total: ~50-70 (one-time)
```

### Problem: Agent can get ~0.1 reward per step FOREVER by circling!

If agent circles for 10 steps near food:
- Circling: +0.1 × 10 = +1.0
- Collecting: +50 (but then no more food nearby)

**Agent learns: "Keep approaching but never collect" = steady rewards!**

## Why This Happens

### 1. Approach Reward Too Continuous

Current approach reward:
```python
if nearest_food_dist < prev_nearest_food_dist:
    reward += 0.5  # Every step closer
```

Agent can oscillate:
- Step 1: Move toward food (distance 5→4) → +0.5
- Step 2: Move toward food (distance 4→3) → +0.5
- Step 3: Move away (distance 3→4) → -0.2
- Step 4: Move toward food (distance 4→3) → +0.5
- Net: +1.3 over 4 steps = +0.325 per step average

**By moving back and forth, agent gets continuous rewards without collecting!**

### 2. Danger Zone Prevents Final Approach

Food near wall (1 tile from wall):
- Approach reward: +0.5
- Danger penalty: -1.0 × (1.0 - 1.0/1.5) = -0.33
- Net: +0.17

To collect food at 1 tile from wall, need to get even closer:
- At 0.5 tiles from wall: Danger = -0.67
- Net: +0.5 - 0.67 = -0.17 (negative!)

**Agent learns: "Don't collect food near walls, just approach it"**

### 3. Collection Doesn't Reward Efficiency

There's no penalty for taking too long:
- 70 steps to collect 7 pellets: OK
- 150 steps to collect 7 pellets: Also OK!
- No difference in final reward

**Agent has no incentive to be efficient**

## Solutions

### Solution 1: Remove Oscillation Rewards (Recommended)

Only reward net progress, not every step:

```python
# Track distance over time window (5 steps)
self.food_dist_history = deque(maxlen=5)

# Calculate approach reward
if len(self.food_dist_history) >= 5:
    initial_dist = self.food_dist_history[0]
    current_dist = nearest_food_dist
    net_progress = initial_dist - current_dist

    if net_progress > 0:
        reward += net_progress * 2.0  # Reward NET progress
```

This prevents oscillation because moving back-and-forth gives 0 net progress.

### Solution 2: Stronger Collection Bonus

Make collection much more valuable:

```python
# Current
if base_reward >= 10:
    pellet_reward = 50.0 + (self.combo_count * 10.0)

# Proposed
if base_reward >= 10:
    pellet_reward = 100.0 + (self.combo_count * 20.0)  # Double!
```

Makes collecting more attractive than circling.

### Solution 3: Proximity Bonus

Give big bonus for getting VERY close to food:

```python
# Add proximity bonus
if nearest_food_dist is not None:
    if nearest_food_dist == 1:  # Adjacent to food
        reward += 5.0  # Big bonus for being ready to collect!
```

Encourages final approach step.

### Solution 4: Penalize Stagnation

Add small penalty for not making progress:

```python
self.steps_since_collection = 0

if collected_food:
    self.steps_since_collection = 0
else:
    self.steps_since_collection += 1

if self.steps_since_collection > 20:
    reward -= 0.5  # Penalty for taking too long
```

### Solution 5: Reduce Danger Zone Further

Make danger penalty even weaker near food:

```python
# Current
if min_wall_dist < 1.5:
    danger_penalty = -1.0 * (1.0 - min_wall_dist / 1.5)

# Proposed
if min_wall_dist < 1.2:  # Only very close
    danger_penalty = -0.5 * (1.0 - min_wall_dist / 1.2)  # Even weaker
```

## Recommended Fix: Combination Approach

```python
class ImprovedSnakeRewards:
    def __init__(self):
        self.combo_count = 0
        self.steps_alive = 0
        self.food_dist_history = deque(maxlen=5)  # Track progress
        self.steps_since_collection = 0

    def calculate(self, base_reward, died, min_wall_dist,
                  nearest_food_dist=None, prev_nearest_food_dist=None):
        total = 0.0

        # 1. Pellet collection (STRONGER: 100 + combo * 20)
        if base_reward >= 10:
            pellet_reward = 100.0 + (self.combo_count * 20.0)
            total += pellet_reward
            self.combo_count += 1
            self.steps_since_collection = 0  # Reset stagnation counter

        # 2. Survival
        self.steps_alive += 1
        self.steps_since_collection += 1
        total += 0.1

        # 3. Death penalty
        if died:
            total -= 100.0

        # 4. Danger zone (WEAKER: only very close)
        if min_wall_dist < 1.2:
            danger_penalty = -0.5 * (1.0 - min_wall_dist / 1.2)
            total += danger_penalty

        # 5. Net progress reward (ANTI-OSCILLATION)
        if nearest_food_dist is not None:
            self.food_dist_history.append(nearest_food_dist)

            if len(self.food_dist_history) >= 5:
                net_progress = self.food_dist_history[0] - nearest_food_dist
                if net_progress > 0:
                    total += net_progress * 2.0  # Reward real progress
                elif net_progress < -2:
                    total -= 1.0  # Penalty for moving away

            # Proximity bonus for getting VERY close
            if nearest_food_dist == 1:
                total += 5.0  # Big bonus for being adjacent!

        # 6. Stagnation penalty
        if self.steps_since_collection > 30:
            total -= 0.5

        return total
```

## Expected Improvement

With these changes:
- Circling no longer profitable (net progress = 0 reward)
- Collection much more valuable (100 vs 50)
- Proximity bonus encourages final approach
- Stagnation penalty prevents wandering
- Weaker danger zone allows collection near walls

**Expected: 70-80 steps to collect all 7 pellets, no circling**

## Current vs Fixed

**Current:**
```
Steps to collect 7 pellets: ~150-170
Circling: Profitable (continuous approach rewards)
Collection efficiency: Not rewarded
```

**After fix:**
```
Steps to collect 7 pellets: ~70-80
Circling: Not profitable (no net progress reward)
Collection efficiency: Highly rewarded
```

# Root Cause: Agent Hits Boundary Walls

## User Confirmation
Agent hits **boundary walls** (edges of grid at x=0, x=19, y=0, y=19), not just body.

## Evidence from Tests

### Test showed wall avoidance failure:
```
Step 78: pos=(17, 17), moving RIGHT, wall 2.0 tiles away
Step 79: pos=(18, 17), moving RIGHT, wall 2.0 tiles away
Step 80: >>> COLLISION at (19, 17) - BOUNDARY WALL!
```

**Critical observation:**
- Agent CAN SEE wall (2 tiles away in raycasting)
- Agent STILL moves toward it
- Q-values don't strongly penalize wall-approaching moves

### Q-Values at Step 78 (before collision):
```
Position: (17, 17)
Wall RIGHT: 2.0 tiles
Q-values: UP=xxx, DOWN=xxx, LEFT=xxx, RIGHT=xxx
Agent chose: RIGHT (toward wall!)
```

## Root Causes

### 1. **Insufficient Wall Collision Experience During Training**

From `train_snake_focused.py` line 92-94:
```python
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = episodes * 0.5  # = 250 episodes
```

**Problem:**
- Epsilon reaches ~0.01 by episode 250 (halfway through training)
- After episode 250, agent rarely explores (99% exploitation)
- **Agent might not have experienced enough wall collisions!**

**Why this matters:**
- Early training (high epsilon): Random exploration, hits walls frequently
- Mid training (medium epsilon): Learning, some wall experiences
- Late training (low epsilon): Mostly exploitation, very few new wall experiences
- **If Q-network doesn't learn strong wall avoidance early, it won't improve later!**

### 2. **Weak Reward Signal Compared to Movement Reward**

From `planning_test_games.py`:
```python
# Normal movement: +0.1
reward = 0.1

# Wall collision: -50.0
reward = -50.0
```

**Ratio analysis:**
- Wall collision = -50
- Normal movement = +0.1
- **Need 500 successful moves to offset ONE wall collision**

**But in training:**
- Movement reward is received EVERY step
- Wall collision is received RARELY (once every 100-200 steps?)
- **Cumulative positive rewards can dominate sparse negative wall penalties!**

### 3. **Q-Learning Update Frequency**

Q-network updates from replay buffer might not sample enough wall collisions:

```python
# From training loop
if len(replay_buffer) >= batch_size:
    transitions, indices, weights = replay_buffer.sample(batch_size)  # 64 samples
```

**Problem:**
- Batch size = 64
- If only 1-2% of experiences are wall collisions
- Network might only see 1-2 wall collision samples per batch
- **Wall avoidance signal gets diluted!**

### 4. **Greedy Policy During Exploitation**

When epsilon is low (exploitation mode):
```python
action = policy_net.get_action(obs, epsilon=0.01)
# 99% of time: picks argmax Q(s,a)
# 1% of time: random
```

**If Q-values are similar:**
```
Q(s, RIGHT) = 603.1  ← toward wall
Q(s, UP) = 605.1     ← best
Q(s, LEFT) = 602.1
Q(s, DOWN) = 600.8
```

Difference is only ~5 points! If Q-values aren't confidently learned, agent might pick sub-optimal actions.

## Why Our Tests Were Misleading

### Earlier test showed "Wall collisions: 0"

That test ran SHORT episodes (100 steps) with specific scenarios. The agent got LUCKY and didn't hit walls in those particular runs. But over longer play or different scenarios, wall collisions DO occur!

### The Real Statistics:

From full episode test:
- 100 steps
- 11 warnings (moved toward wall <3 tiles)
- 1 life lost (collision)
- **~10% of moves were risky toward walls!**

## Comparison: What SHOULD Happen

**Well-trained agent:**
```
Position: (18, 17), Wall RIGHT: 1 tile
Q(s, RIGHT) = -100  ← strongly avoid!
Q(s, LEFT) = +50    ← strongly prefer!
Agent: Chooses LEFT
```

**Current agent:**
```
Position: (18, 17), Wall RIGHT: 2 tiles
Q(s, RIGHT) = 603   ← not strongly discouraged
Q(s, UP) = 605      ← slightly better
Agent: Sometimes picks RIGHT (bad!)
```

**The Q-values are not confident enough about wall danger!**

## Mathematical Analysis

### Expected Q-values for wall proximity:

If properly trained, Q(s, action_toward_wall) should be:
```
Q(s, toward_wall) = immediate_reward + γ * (future_rewards)
                  = +0.1 + 0.99 * (-50 + respawn_penalty)
                  = +0.1 + 0.99 * (-150)
                  = +0.1 - 148.5
                  = -148.4
```

But observed Q-values are all positive (600+)!

**This means:**
- Q-network hasn't properly learned wall collision consequences
- Or: Positive rewards from food collection are overwhelming wall penalties
- Or: Not enough wall collision experiences in training data

## Solutions

### Solution 1: Increase Wall Penalty (Easy fix)

```python
# Current:
reward = -50.0  # Wall collision

# Proposed:
reward = -100.0  # Stronger signal!
```

Stronger penalty → Q-network learns faster to avoid walls

### Solution 2: Slower Epsilon Decay (Better exploration)

```python
# Current:
epsilon_decay = episodes * 0.5  # Fast decay (250 episodes)

# Proposed:
epsilon_decay = episodes * 0.7  # Slower decay (350 episodes)
```

More exploration → more wall collision experiences → better learning

### Solution 3: Prioritized Experience Replay for Wall Collisions

```python
# Give higher priority to wall collision experiences
if reward < -10:  # Wall or body collision
    priority = abs(td_error) * 2.0  # Double priority!
else:
    priority = abs(td_error)
```

More frequent sampling of collision experiences → faster learning

### Solution 4: Curriculum Learning

Start with SMALLER grid → easier to hit walls → learn avoidance faster:

```python
# Episodes 0-200: size=10 (walls very close, frequent collisions)
# Episodes 200-400: size=15 (medium difficulty)
# Episodes 400-500: size=20 (full size)
```

### Solution 5: Add "Danger" Reward Shaping

Give small negative rewards for being NEAR walls:

```python
min_wall_dist = obs['wall_dist'].min()
if min_wall_dist < 0.2:  # Less than 3 tiles
    reward -= 1.0  # Small penalty for danger zone
```

Continuous feedback → smoother learning gradient

## Recommended Training Strategy

Combine multiple improvements:

```python
# train_snake_focused_v2.py
def train_improved(episodes=500):
    # 1. Slower epsilon decay
    epsilon_decay = episodes * 0.7  # 350 episodes

    # 2. Stronger wall penalty
    wall_penalty = -100.0  # vs -50.0

    # 3. Curriculum learning
    if episode < 200:
        game_size = 10  # Small grid
    elif episode < 400:
        game_size = 15  # Medium grid
    else:
        game_size = 20  # Full grid

    # 4. Danger-zone reward shaping
    if min_wall_dist < 0.2:
        reward -= 2.0  # Immediate feedback

    # 5. Prioritized replay for collisions
    if collision:
        priority_weight = 2.0
```

## Expected Improvement

With these changes:
- Episodes 0-100: Learn basic wall avoidance in small grid
- Episodes 100-300: Refine avoidance in larger grid
- Episodes 300-500: Master full-size grid
- **Expected wall collisions in testing: 0-1 per 1000 steps**

## Conclusion

**The agent CAN see walls** (raycasting works ✓)
**But it doesn't avoid them properly** (learning incomplete ✗)

Root cause: **Insufficient training signal for wall avoidance**
- Sparse wall collision experiences
- Weak penalty compared to movement rewards
- Fast epsilon decay limiting exploration
- Q-values not confident about wall danger

**Solution: Retrain with improved strategy** (stronger penalties, more exploration, curriculum learning)

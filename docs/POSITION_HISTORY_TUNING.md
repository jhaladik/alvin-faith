# Position History Tuning - Environment-Specific Anti-Oscillation

## The Challenge

The "fake tail" position history prevents oscillation, but the **optimal length depends on the environment**!

## Problem Discovery

When testing the snake agent on the dungeon game, we discovered the agent **getting stuck in corners** at lower difficulty levels.

### Root Cause

**In tight corridors:**
```
Step 1: Agent at A
Step 2: Move to B → history = [A, B]
Step 3: Move to C → history = [A, B, C]
Step 4: Hit dead end at C
Step 5: Want to backtrack to B? ❌ BLOCKED (B is in history!)
Step 6: Want to go back to A? ❌ BLOCKED (A is in history!)
Result: DEADLOCK - Agent is stuck!
```

With 3-position history, the agent **cannot escape dead ends**.

## Solution: Configurable History Length

Different environments need different history lengths:

### PacMan (Open Maze)
```python
SimplePacManGame(size=20, num_pellets=30)
# Uses 3-position history by default
```

**Why 3?**
- Open spaces with multiple paths
- Lots of room to navigate
- 3 positions prevent A→B→A→B oscillation
- Unlikely to encounter dead ends

### Dungeon (Tight Corridors)
```python
SimpleDungeonGame(size=20, num_treasures=3, history_length=2)
# Uses 2-position history (configurable)
```

**Why 2?**
- Tight corridors with dead ends
- Needs tactical retreats
- 2 positions prevent A→B→A immediate reversal
- But allows A→B→C→B backtracking when needed

### Snake (Original Training)
```python
# Snake has actual tail - no fake tail needed!
# Body naturally prevents backtracking
```

## History Length Comparison

### Length = 1 (Immediate Previous Only)
```
Blocks: A → B → A
Allows: A → B → C → B
```
✅ Maximum flexibility
✅ Allows all tactical retreats
⚠️ Minimal oscillation protection

### Length = 2 (Short Memory)
```
Blocks: A → B → A, also blocks A → B → C → A
Allows: A → B → C → B, A → B → C → D → C
```
✅ Good balance
✅ Prevents simple oscillation
✅ Allows tactical retreats
👍 **BEST FOR DUNGEONS**

### Length = 3 (Strict Memory)
```
Blocks: A → B → A, A → B → C → A, A → B → C → B
Allows: A → B → C → D → C
```
✅ Strong oscillation protection
❌ Can cause deadlocks in corridors
👍 **BEST FOR OPEN MAZES (PacMan)**

### Length = 4+ (Very Strict)
```
Blocks almost all backtracking
```
❌ Too restrictive
❌ High chance of deadlocks
❌ Not recommended

## Implementation

### Dungeon Game
```python
class SimpleDungeonGame:
    def __init__(self, size=20, num_treasures=3, enemy_level=0,
                 max_steps=500, history_length=2):
        self.history_length = history_length  # Configurable!

    def step(self, action):
        # Update position history
        self.position_history.append(self.player_pos)
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
```

### PacMan Game
```python
class SimplePacManGame:
    def __init__(self, size=20, num_pellets=30, ghost_level=0, max_steps=500):
        # PacMan uses fixed 3-position history (open maze)

    def step(self, action):
        # Update position history (keep last 3 positions)
        self.position_history.append(self.pacman_pos)
        if len(self.position_history) > 3:
            self.position_history.pop(0)
```

## Results

### Before Fix (Length = 3)
```
Dungeon Test Results:
- Agent gets stuck in corners
- 0% win rate at level 0 (no enemies!)
- Deadlock in tight corridors
```

### After Fix (Length = 2)
```
Dungeon Test Results:
- Agent can escape dead ends
- Can navigate tight corridors
- Tactical retreats possible
- Should see improved performance
```

## General Guidelines

**Choose history length based on environment:**

1. **Open spaces** (PacMan, large rooms): `length = 3`
   - Multiple paths available
   - Strong oscillation prevention needed
   - Low risk of deadlocks

2. **Tight corridors** (Dungeons, mazes): `length = 2`
   - Dead ends common
   - Tactical retreats necessary
   - Moderate oscillation prevention sufficient

3. **Mixed environments**: `length = 2`
   - Better to allow retreats than risk deadlocks
   - Agent learns to avoid oscillation over time

4. **No restrictions**: `length = 1` or disabled
   - Maximum flexibility
   - Suitable for environments with actual constraints (snake body)
   - Agent must learn anti-oscillation behavior

## Key Insight

**Environment topology matters!**

The same anti-oscillation mechanism that works perfectly in open spaces can cause deadlocks in tight corridors. Always tune position history length to match the environment's navigation constraints.

---

**Test Command:**
```bash
# Dungeon with 2-position history (default)
python test_zero_shot_dungeon.py --model checkpoints/snake_improved_20251121_150114_policy.pth --enemy-level 0 --speed 10

# PacMan with 3-position history (fixed)
python test_zero_shot_pacman.py --model checkpoints/snake_improved_20251121_150114_policy.pth --ghost-level 0 --speed 10
```

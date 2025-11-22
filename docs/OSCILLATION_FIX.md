# Oscillation Fix - Preventing Back-and-Forth Movement

## The Problem

When the snake agent played PacMan, it was **oscillating** (moving back and forth between two positions).

### Root Cause

**In Snake (Training):**
- Agent has a growing tail
- Can't move backward → would hit own body
- This constraint prevented oscillation naturally

**In PacMan (Zero-Shot Transfer):**
- No tail/body
- CAN move backward freely
- Agent never learned "don't oscillate" without the body constraint
- Result: Back-and-forth movement

### Why It Happens

1. Agent sees pellet at distance 5
2. Moves toward it (distance becomes 4)
3. Q-values might prefer moving back (due to exploration or local optima)
4. Without body blocking backward movement, agent oscillates
5. Net progress = 0

## The Solution: "Fake Tail" - Position History

We give PacMan a **short memory** of recent positions that acts like a snake's tail.

### Implementation

```python
class SimplePacManGame:
    def reset(self):
        # Track last 3 positions
        self.position_history = [self.pacman_pos]

    def step(self, action):
        # After moving successfully
        self.position_history.append(self.pacman_pos)
        if len(self.position_history) > 3:
            self.position_history.pop(0)  # Keep only last 3

    def _get_game_state(self):
        # Recent positions act like snake body
        fake_tail = self.position_history[:-1]

        return {
            'snake_body': fake_tail,  # Observer treats these as obstacles!
            ...
        }
```

### How It Works

1. **Track Movement**: Store last 3 positions
2. **Fake Tail**: Expose history as `snake_body` in game state
3. **Observer Sees It**: Raycasting detects fake tail as obstacles
4. **Agent Avoids**: Can't move back to recent positions
5. **No Oscillation**: Forced to make forward progress

### Parameters

**History Length: 3 positions**
- Too short (1-2): Can still oscillate
- Just right (3): Prevents immediate backtracking
- Too long (5+): Overly restrictive, limits movement

### Example

```
Step 1: PacMan at (10, 10) → history = [(10, 10)]
Step 2: Move RIGHT to (11, 10) → history = [(10, 10), (11, 10)]
Step 3: Move RIGHT to (12, 10) → history = [(10, 10), (11, 10), (12, 10)]

Game state shows:
- agent_pos = (12, 10)
- snake_body = [(10, 10), (11, 10)]  ← Fake tail!

If agent tries to move LEFT:
- Would go to (11, 10)
- But (11, 10) is in snake_body!
- Observer sees this as obstacle
- Agent won't choose this action
- Result: Forward progress maintained
```

## Benefits

✅ **Prevents oscillation** - Can't go back to recent positions
✅ **Mimics training constraint** - Similar to snake's body
✅ **Zero modification to agent** - Works with existing checkpoint
✅ **Minimal restriction** - Only blocks last 3 positions
✅ **Natural behavior** - Agent acts like it has a short tail

## Alternative Solutions (Not Used)

### 1. Reward Penalty for Reversing
```python
# Penalize moving to previous position
if new_pos == self.prev_pos:
    reward -= 1.0
```
❌ Doesn't prevent oscillation, just penalizes it

### 2. Net Progress Tracking (Like Snake Training)
```python
# Only reward net progress over 5 steps
# (This is what we did in snake training)
```
❌ Requires retraining, doesn't work zero-shot

### 3. Directional Momentum
```python
# Prefer continuing current direction
```
❌ Complex, changes agent behavior too much

### 4. Hard Block Previous Position
```python
# Treat previous position as wall
```
✅ **This is what we implemented!** (generalized to 3 positions)

## Test Results Expected

**Before Fix:**
```
Agent behavior: A → B → A → B → A → B (oscillating)
Progress: Zero net movement
Pellets collected: Very few
```

**After Fix:**
```
Agent behavior: A → B → C → D (forward movement)
Progress: Continuous exploration
Pellets collected: Much more!
```

## Try It Now

```bash
python test_zero_shot_pacman.py --model checkpoints/snake_improved_20251121_150114_policy.pth --ghost-level 0 --speed 8
```

The agent should now move forward consistently without oscillating!

## Technical Details

### Observer Integration
The fake tail integrates seamlessly with the existing observer:

```python
# In expanded_temporal_observer.py (line 249-251)
if snake_body and (px, py) in snake_body:
    wall_dist = min(wall_dist, step)
    break  # Treat as obstacle
```

The observer already had code to detect snake body - we just provide recent positions as the "body"!

### Why This Is Clever

This solution:
1. Reuses existing infrastructure (snake body detection)
2. Requires zero changes to agent or observer
3. Only modifies the game environment
4. Makes PacMan behave more like Snake (which agent knows!)

**The constraint that prevented oscillation in Snake training now applies to PacMan too.**

# Visual Game Update - Anti-Circling Edition

## Summary

The `expanded_faith_visual_games.py` has been updated to match the new training system with anti-circling features and progressive difficulty.

## What Was Updated

### 1. Imports
- Added `EnhancedSnakeGame` import
- Added `deque` from collections for reward tracking

### 2. SnakeTrainingMatchedRewardSystem (Completely Rewritten)

**Now matches train_snake_improved.py EXACTLY:**

#### Old System (Circling-Prone):
```python
- Pellet: 50 + combo*10
- Danger zone: -1.0 when <1.5 tiles from wall
- Approach: +0.5 toward, -0.2 away (per step)
- No stagnation penalty
- No proximity bonus
```

#### New System (Anti-Circling):
```python
- Pellet: 100 + combo*20 (2× stronger!)
- Danger zone: -0.5 when <1.2 tiles (weaker, allows collection)
- NET progress: Rewards real progress over 5-step window (anti-circling!)
- Proximity bonus: +5 when adjacent to food
- Stagnation penalty: -0.5 after 30 steps without collection
```

**Key Anti-Circling Feature:**
- Uses `deque(maxlen=5)` to track food distance history
- Only rewards NET progress over 5 steps, not per-step movement
- Oscillating around food gives 0 net progress = 0 reward
- Moving back-and-forth no longer profitable

### 3. EnhancedSnakeGame Integration

**Game Instantiation:**
```python
self.current_game = EnhancedSnakeGame(
    size=20,
    initial_pellets=7,      # Start with 7 food
    max_pellets=12,         # Can increase to 12 as score grows
    food_timeout=150,       # Food disappears after 150 steps
    obstacle_level=2        # Large cross obstacles in center
)
```

**Progressive Difficulty Features:**
- Food count increases as score improves: 7→8→9→...→12
- Food disappears after 150 steps (creates urgency)
- Central obstacles (large cross pattern) for navigation challenge

### 4. Rendering Updates

**Central Obstacles:**
- Gray squares drawn for central obstacles
- Obstacles act as walls (can't pass through)
- Visible in game display

**Info Panel Updates:**
- Game name changed to "SNAKE (Enhanced)"
- Shows progressive food count
- Shows target food count (progressive)
- Shows obstacle count (if present)

Example display:
```
SNAKE (Enhanced)
Score: 5
Steps: 42
Reward: 523.4
Lives: 3
Length: 8
Food: 9         ← Current food on board
Target: 10      ← Progressive target
Obstacles: 10   ← Central cross count
```

### 5. Reward Calculation Updates

The reward breakdown now includes:
```python
breakdown = {
    'env': env_reward,           # Base game reward
    'pellet': 0.0,               # Collection: 100 + combo*20
    'survival': 0.0,             # +0.1 per step
    'death': 0.0,                # -100 on death
    'danger': 0.0,               # -0.5 when <1.2 from wall
    'net_progress': 0.0,         # NET progress over 5 steps
    'proximity': 0.0,            # +5 when adjacent
    'stagnation': 0.0            # -0.5 after 30 steps
}
```

## Testing the Visual Game

To test the updated visual game:

```bash
python src/expanded_faith_visual_games.py --checkpoint checkpoints/snake_improved_[timestamp]_policy.pth
```

Press `1` to select Snake game.

### What to Observe

1. **No Circling**: Agent should approach food directly
2. **Progressive Food**: More food appears as score increases
3. **Obstacles**: Gray cross pattern in center
4. **Food Timeout**: Food disappears if not collected quickly
5. **Efficient Collection**: Much faster than before (50-80 steps vs 150-170)

### Visual Indicators

- **Snake head color**:
  - Cyan: Reactive action
  - Magenta: Faith-based action
  - Cyan with planning: Planning action

- **Central obstacles**: Gray squares forming cross pattern
- **Food**: Red circles (disappear after 150 steps if not collected)
- **Boundaries**: Dark green walls

## Compatibility Notes

- Visual game rewards now **EXACTLY match** training rewards
- Q-values from trained model are now meaningful in visual game
- Agent behavior will be accurate to training
- No more confusion from reward mismatch

## Expected Behavior Changes

### Before (Old Visual Game):
```
- Agent circles around food
- Long episode lengths (150-170 steps)
- Doesn't approach food near walls
- Q-values mismatched (trained on 50, saw 10)
```

### After (Updated Visual Game):
```
- Agent collects food directly
- Short episode lengths (50-80 steps)
- Collects food near walls confidently
- Q-values matched (trained on 100, sees 100)
```

## Summary of Changes

| Component | Old | New | Impact |
|-----------|-----|-----|--------|
| Pellet reward | 50 + combo×10 | 100 + combo×20 | 2× stronger incentive |
| Approach reward | Per-step +0.5 | NET progress only | Prevents circling |
| Danger zone | -1.0 @ <1.5 | -0.5 @ <1.2 | Allows wall collection |
| Proximity | None | +5 @ adjacent | Encourages final approach |
| Stagnation | None | -0.5 @ 30 steps | Prevents wandering |
| Progressive food | No | Yes (7→12) | Increasing challenge |
| Food timeout | No | Yes (150 steps) | Creates urgency |
| Obstacles | No | Yes (level 2) | Navigation challenge |

The visual game is now fully aligned with the training system!

# Faith Agent: Transfer Learning Improvement Action Plan

## Current Status (Baseline)

**What Works:**
- ✓ Snake → Pac-Man (easy): 40% win rate
- ✓ Spatial reasoning transfers
- ✓ Dense reward collection

**What Doesn't Work:**
- ✗ Snake → Pac-Man (chase): 0% win rate
- ✗ Snake → Dungeon: 0% win rate
- ✗ Sparse reward exploration
- ✗ Active threat avoidance

## Priority Action Items

### Priority 1: Fix Dungeon Transfer (Sparse Rewards)
**Target**: 0% → 30% win rate in Dungeon Level 0

#### Action 1.1: Add Exploration Bonus to Training
**File**: `src/core/enhanced_snake_game.py` (or create new variant)

```python
class ExplorationSnakeGame(SnakeGame):
    def __init__(self, *args, exploration_bonus=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration_bonus = exploration_bonus
        self.visited_cells = set()

    def step(self, action):
        state, reward, done = super().step(action)

        # Add exploration bonus
        if self.snake_pos not in self.visited_cells:
            reward += self.exploration_bonus
            self.visited_cells.add(self.snake_pos)

        return state, reward, done
```

**Expected Impact**: Agent learns to explore systematically, should improve dungeon performance.

#### Action 1.2: Curriculum Training with Reward Sparsity
**File**: Create `src/train_curriculum.py`

```python
# Training phases
phases = [
    # Phase 1: Dense rewards (current)
    {'food_count': 15, 'episodes': 500},

    # Phase 2: Medium sparsity
    {'food_count': 8, 'episodes': 500},

    # Phase 3: Sparse rewards (dungeon-like)
    {'food_count': 3, 'episodes': 500},
]

for phase in phases:
    game = SnakeGame(num_food=phase['food_count'])
    train(game, episodes=phase['episodes'])
```

**Expected Impact**: Agent learns to handle sparse rewards gradually.

#### Action 1.3: Adjust Position History for Maze Navigation
**File**: `src/core/simple_dungeon_game.py`

**Already implemented!** Dungeon uses `history_length=2` vs Pac-Man's 3.
- Test if reducing to `history_length=1` improves exploration
- May need context-aware history based on corridor detection

**Test Command**:
```bash
# Test with different history lengths
python test_zero_shot_dungeon.py --model checkpoints/snake_improved_20251121_150114_policy.pth
```

### Priority 2: Enable Active Threat Avoidance (Chasing Ghosts)
**Target**: 0% → 20% win rate in Pac-Man Level 4 (chase)

#### Action 2.1: Add Predator to Snake Training
**File**: Create `src/core/snake_with_predator.py`

```python
class SnakeWithPredatorGame(SnakeGame):
    def __init__(self, *args, predator_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.predator_enabled = predator_enabled
        if predator_enabled:
            self.predator_pos = self._spawn_predator()

    def _spawn_predator(self):
        # Spawn predator far from snake
        while True:
            pos = (random.randint(2, self.size-3),
                   random.randint(2, self.size-3))
            if abs(pos[0] - self.snake_pos[0]) > 5 and \
               abs(pos[1] - self.snake_pos[1]) > 5:
                return pos

    def _move_predator(self):
        # Simple chase behavior (like Pac-Man ghosts)
        sx, sy = self.snake_pos
        px, py = self.predator_pos

        # Move toward snake
        possible_moves = []
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            new_pos = (px+dx, py+dy)
            if self._is_valid_pos(new_pos):
                dist = abs(new_pos[0]-sx) + abs(new_pos[1]-sy)
                possible_moves.append((dist, new_pos))

        if possible_moves:
            possible_moves.sort()
            self.predator_pos = possible_moves[0][1]

    def step(self, action):
        # Move snake
        state, reward, done = super().step(action)

        # Move predator
        if self.predator_enabled and not done:
            self._move_predator()

            # Check collision
            if self.snake_pos == self.predator_pos:
                reward = -50.0
                done = True

        return state, reward, done
```

**Expected Impact**: Agent learns "flee from approaching threat" behavior.

#### Action 2.2: Augment Observer to Detect Threat Direction
**File**: `src/core/expanded_temporal_observer.py`

**Already working!** Ghosts are detected as 'entities' in rays.
- Verify ray detection includes velocity/direction of threats
- May need "threat level" based on distance and approaching velocity

### Priority 3: Improve Overall Exploration Strategy

#### Action 3.1: Add Spatial Memory Module
**File**: Create `src/spatial_memory.py`

```python
class SpatialMemory:
    """
    Remember visited locations and reward locations
    """
    def __init__(self, grid_size, memory_size=100):
        self.grid_size = grid_size
        self.memory_size = memory_size
        self.visited = {}  # position -> visit_count
        self.rewards_seen = {}  # position -> last_seen_step
        self.current_step = 0

    def update(self, agent_pos, reward_positions):
        self.current_step += 1

        # Update visited
        self.visited[agent_pos] = self.visited.get(agent_pos, 0) + 1

        # Update reward memory
        for pos in reward_positions:
            self.rewards_seen[pos] = self.current_step

        # Prune old memories
        if len(self.visited) > self.memory_size:
            self._prune_visited()

    def get_exploration_bonus(self, pos):
        """Lower visit count = higher bonus"""
        visits = self.visited.get(pos, 0)
        return 1.0 / (1.0 + visits)

    def get_memory_encoding(self, agent_pos, radius=5):
        """
        Encode nearby memory as features
        Returns: [avg_visits_nearby, known_rewards_nearby]
        """
        # Implementation...
        pass
```

**Integration**: Add memory features to observation space.

#### Action 3.2: Hierarchical Options Framework
**File**: Create `src/hierarchical_agent.py`

```python
class HierarchicalAgent:
    """
    High-level: Choose option (Explore, Exploit, Avoid)
    Low-level: Execute movement primitives
    """
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.high_level_policy = OptionPolicy()
        self.current_option = None

    def choose_option(self, state):
        """
        Options:
        - EXPLORE: Go to least-visited area
        - EXPLOIT: Go to known reward location
        - AVOID: Flee from threats
        """
        pass

    def get_action(self, state):
        # High-level decides which option
        if self.current_option is None or self.option_terminated():
            self.current_option = self.choose_option(state)

        # Low-level executes option
        return self.current_option.get_action(state, self.base_agent)
```

## Implementation Roadmap

### Week 1: Quick Wins
1. **Day 1-2**: Implement exploration bonus (Action 1.1)
   - Modify snake game to reward visiting new cells
   - Retrain for 500 episodes
   - Test on dungeon

2. **Day 3-4**: Add predator to snake training (Action 2.1)
   - Create SnakeWithPredatorGame
   - Train for 500 episodes
   - Test on Pac-Man Level 4 (chase)

3. **Day 5**: Run comprehensive tests
   - Compare new checkpoints vs baseline
   - Document improvements

### Week 2: Curriculum Learning
1. **Day 1-3**: Implement curriculum training (Action 1.2)
   - Create training script with phased reward sparsity
   - Train full curriculum (1500 episodes)

2. **Day 4-5**: Test and analyze
   - Test on all difficulty levels
   - Create comparison charts
   - Identify remaining gaps

### Week 3: Advanced Features
1. **Day 1-3**: Implement spatial memory (Action 3.1)
   - Create SpatialMemory module
   - Integrate with observer
   - Retrain agent

2. **Day 4-5**: Test and document
   - Run full test suite
   - Document performance improvements
   - Write research paper draft

### Future Work: Hierarchical RL
- Research option frameworks (Sutton et al.)
- Implement HAM or options
- Train with multiple time-scales

## Testing Protocol

### For Each New Checkpoint

```bash
# 1. Run automated tests
python analyze_transfer_results.py --model <checkpoint> --episodes 20

# 2. Visual verification (spot check)
python test_zero_shot_pacman.py --model <checkpoint> --ghost-level 2
python test_zero_shot_dungeon.py --model <checkpoint> --enemy-level 0

# 3. Compare to baseline
# Record: win_rate, avg_score, exploration coverage
```

### Success Criteria

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| Pac-Man (easy) | 40% | 60% | 80% |
| Pac-Man (chase) | 0% | 20% | 40% |
| Dungeon (no enemy) | 0% | 30% | 60% |
| Dungeon (with enemy) | 0% | 10% | 30% |

## Code Changes Summary

### New Files to Create
```
src/
  core/
    exploration_snake_game.py       # Snake with exploration bonus
    snake_with_predator.py          # Snake with chasing predator
  spatial_memory.py                 # Memory module
  hierarchical_agent.py             # High-level + low-level policies
  train_curriculum.py               # Curriculum training script

tests/
  test_exploration_bonus.py         # Unit tests for exploration
  test_predator_avoidance.py        # Unit tests for predator
  test_spatial_memory.py            # Unit tests for memory
```

### Files to Modify
```
src/
  core/
    expanded_temporal_observer.py   # Add threat direction detection
    simple_dungeon_game.py          # Test different history_length values
  context_aware_agent.py            # Integrate spatial memory features
```

## Experiment Tracking

Create experiment log to track all training runs:

```markdown
# experiments.md

## Experiment 1: Exploration Bonus
- Date: 2025-11-21
- Checkpoint: snake_explore_bonus_v1.pth
- Changes: Added 0.3 exploration bonus
- Results:
  - Pac-Man: 42% (vs 40% baseline)
  - Dungeon: 15% (vs 0% baseline) ✓ IMPROVEMENT!

## Experiment 2: Predator Training
- Date: 2025-11-22
- Checkpoint: snake_with_predator_v1.pth
- Changes: Added chasing predator during training
- Results:
  - Pac-Man (chase): 25% (vs 0% baseline) ✓ IMPROVEMENT!
  - Pac-Man (easy): 35% (vs 40% baseline) - slight regression

## Experiment 3: Curriculum Learning
...
```

## Monitoring & Debugging

### Visualization Tools Needed

1. **Exploration Heatmap**
   ```python
   # Show which cells agent visits during episode
   plt.imshow(visit_counts)
   plt.title('Agent Exploration Pattern')
   ```

2. **Ray Visualization**
   ```python
   # Debug what agent "sees"
   def visualize_rays(state, rays):
       # Draw rays and what they detect
       pass
   ```

3. **Reward Timeline**
   ```python
   # Plot rewards over time to identify sparse reward problem
   plt.plot(episode_rewards)
   plt.title('Reward Distribution')
   ```

## Expected Outcomes

### Optimistic Scenario (All Improvements Work)
- Pac-Man (chase): 0% → 30% win rate
- Dungeon (no enemy): 0% → 50% win rate
- Dungeon (with enemy): 0% → 20% win rate
- **Result**: True foundation agent with robust transfer learning

### Realistic Scenario (Partial Success)
- Pac-Man (chase): 0% → 15% win rate
- Dungeon (no enemy): 0% → 25% win rate
- Dungeon (with enemy): 0% → 5% win rate
- **Result**: Improved transfer, but still game-specific limitations

### Pessimistic Scenario (Limited Improvement)
- Improvements only help in training game (Snake)
- Transfer still poor (<10% improvement)
- **Result**: May need fundamental architecture changes (hierarchical RL)

## Risk Mitigation

### Risk 1: Exploration Bonus Breaks Snake Performance
**Mitigation**:
- Use small bonus (0.1-0.3)
- Test on Snake game first
- Keep baseline checkpoint for comparison

### Risk 2: Predator Training Too Hard
**Mitigation**:
- Start with slow/weak predator
- Gradually increase predator speed/intelligence
- Use curriculum: no predator → slow predator → fast predator

### Risk 3: Changes Don't Transfer
**Mitigation**:
- Test each change independently
- Use ablation studies
- May need to train on multiple games simultaneously

## Questions to Answer Through Experiments

1. **What reward density threshold enables transfer?**
   - Test: 30, 15, 10, 5, 3, 1 rewards
   - Find minimum for >20% success

2. **Does exploration bonus improve transfer?**
   - Compare: with vs without exploration bonus
   - Measure exploration coverage

3. **Can predator training enable threat avoidance?**
   - Train with predator
   - Test on chasing ghosts
   - Measure: flee behavior, survival time

4. **Is position history helping or hurting?**
   - Test history_length: 0, 1, 2, 3, 5
   - Measure: oscillation vs exploration

5. **Does curriculum learning improve sparse reward transfer?**
   - Compare: direct training vs curriculum
   - Measure: dungeon win rate, treasure collection

## Success Metrics

Track these metrics for every experiment:

```python
metrics = {
    'win_rate': 0.0,              # % of episodes won
    'avg_score': 0.0,             # Average reward collected
    'exploration_coverage': 0.0,   # % of map visited
    'avg_steps_to_reward': 0,     # Steps between rewards
    'death_causes': {},           # wall, timeout, enemy, etc.
    'transfer_gap': 0.0,          # vs specialist performance
}
```

## Long-Term Vision

**End Goal**: A foundation agent that can:
1. Transfer to ANY grid-based navigation game
2. Explore efficiently even with sparse rewards
3. Avoid both static and dynamic threats
4. Learn from minimal experience in new environments
5. Exhibit systematic problem-solving strategies

**This is AI research, not just engineering!** Document everything for publication.

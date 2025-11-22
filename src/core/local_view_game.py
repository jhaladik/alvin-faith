"""
Local View Collector Game - Moving Perspective

A game where the agent navigates a larger world with limited visibility.
- Collect coins scattered across the map
- Avoid patrolling enemies (moving walls)
- Camera follows agent (local viewport)

Progressive difficulty:
- Level 0: No enemies, just collect coins
- Level 1: 2 patrolling enemies (random movement)
- Level 2: 3 patrolling enemies (random movement)
- Level 3: 4 patrolling enemies (smarter patrol)
- Level 4+: 5 enemies with occasional chasing behavior
"""
import random


class LocalViewGame:
    """
    Game with moving perspective and local visibility

    Key features:
    - Large world (40x40 or configurable)
    - Agent sees limited area (viewport)
    - Enemies patrol as moving obstacles
    - Medium reward density (20 coins)
    """

    def __init__(self, world_size=40, num_coins=20, enemy_level=0,
                 max_steps=800, viewport_size=20):
        """
        Args:
            world_size: Full world grid size (agent navigates this)
            num_coins: Number of coins to collect
            enemy_level: 0=none, 1=2 enemies, 2=3 enemies, 3=4 enemies, 4+=5 smart
            max_steps: Maximum steps per episode
            viewport_size: Size of observable area (for game state, observer has its own range)
        """
        self.world_size = world_size
        self.num_coins = num_coins
        self.enemy_level = enemy_level
        self.max_steps = max_steps
        self.viewport_size = viewport_size
        self.reset()

    def reset(self):
        """Reset game state"""
        # Agent starts at center of world
        center = self.world_size // 2
        self.agent_pos = (center, center)

        # Create world boundaries (walls around edge)
        self.walls = self._create_world_walls()

        # Add some obstacles in world (optional - makes it more interesting)
        self.walls.update(self._create_obstacles())

        # Place coins scattered across world (avoiding walls)
        self.coins = self._place_coins()

        # Create patrolling enemies based on level
        self.enemies = self._create_enemies()

        # Game state
        self.score = 0
        self.steps = 0
        self.done = False
        self.lives = 3
        self.total_collected = 0

        # Movement history (prevent oscillation)
        self.position_history = [self.agent_pos]

        return self._get_game_state()

    def _create_world_walls(self):
        """Create boundary walls around world"""
        walls = set()

        # Boundary
        for i in range(self.world_size):
            walls.add((0, i))
            walls.add((i, 0))
            walls.add((self.world_size - 1, i))
            walls.add((i, self.world_size - 1))

        return walls

    def _create_obstacles(self):
        """Create some obstacles in world (patches of walls)"""
        obstacles = set()

        # Create 5-8 obstacle patches
        num_patches = random.randint(5, 8)

        for _ in range(num_patches):
            # Random patch location
            cx = random.randint(5, self.world_size - 6)
            cy = random.randint(5, self.world_size - 6)

            # Random patch size (2x2 to 4x4)
            size = random.randint(2, 4)

            # Add patch
            for dx in range(size):
                for dy in range(size):
                    pos = (cx + dx, cy + dy)
                    # Don't place near center (spawn area)
                    center = self.world_size // 2
                    if abs(pos[0] - center) > 5 or abs(pos[1] - center) > 5:
                        obstacles.add(pos)

        return obstacles

    def _place_coins(self):
        """Place coins randomly across world (avoiding walls and spawn)"""
        coins = set()
        center = self.world_size // 2

        # Safe spawn zone (5x5 around center)
        safe_zone = set()
        for i in range(-2, 3):
            for j in range(-2, 3):
                safe_zone.add((center + i, center + j))

        attempts = 0
        max_attempts = 10000

        while len(coins) < self.num_coins and attempts < max_attempts:
            x = random.randint(2, self.world_size - 3)
            y = random.randint(2, self.world_size - 3)
            pos = (x, y)

            # Valid position: not in walls, spawn, or existing coins
            if (pos not in safe_zone and
                pos not in coins and
                pos not in self.walls):
                coins.add(pos)

            attempts += 1

        return coins

    def _create_enemies(self):
        """Create patrolling enemies based on difficulty level"""
        enemies = []

        if self.enemy_level == 0:
            return enemies  # No enemies

        # Number of enemies based on level
        num_enemies = min(self.enemy_level + 1, 5)  # 1-5 enemies

        # Enemy spawn positions (spread around world)
        center = self.world_size // 2
        spawn_positions = [
            (center - 10, center - 10),  # Top-left
            (center + 10, center - 10),  # Top-right
            (center - 10, center + 10),  # Bottom-left
            (center + 10, center + 10),  # Bottom-right
            (center, center - 15),       # Top-center
        ]

        for i in range(min(num_enemies, len(spawn_positions))):
            enemy = {
                'pos': spawn_positions[i],
                'direction': random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)]),
                'behavior': 'patrol' if self.enemy_level < 4 else 'smart_patrol',
                'id': i,
                'patrol_steps': 0
            }
            enemies.append(enemy)

        return enemies

    def _get_game_state(self):
        """
        Get current game state (relative to viewport for rendering,
        but observer will use rays from agent position)
        """
        # Enemy entities (observer expects dicts with 'pos' key)
        enemy_entities = [{'pos': e['pos'], 'type': 'enemy'} for e in self.enemies]

        # Use position history as "fake tail" to prevent backtracking
        fake_tail = self.position_history[:-1] if len(self.position_history) > 1 else []

        # Return global positions (observer will handle ray casting)
        return {
            'agent_pos': self.agent_pos,
            'walls': self.walls,
            'rewards': list(self.coins),
            'entities': enemy_entities,  # Enemies are entities to avoid
            'snake_body': fake_tail,     # Recent positions prevent oscillation
            'grid_size': (self.world_size, self.world_size),  # Full world size
            'score': self.score,
            'done': self.done,
            # Additional info for rendering
            'viewport_center': self.agent_pos,
            'viewport_size': self.viewport_size
        }

    def _is_valid_position(self, pos):
        """Check if position is valid (not wall, in bounds)"""
        x, y = pos
        return (0 <= x < self.world_size and
                0 <= y < self.world_size and
                pos not in self.walls)

    def _move_enemy(self, enemy):
        """Move enemy based on behavior"""
        ex, ey = enemy['pos']

        if enemy['behavior'] == 'patrol':
            # Simple patrol: Move in direction, change on collision
            # 85% continue, 15% random turn
            if random.random() < 0.15 or enemy['patrol_steps'] > 10:
                enemy['direction'] = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                enemy['patrol_steps'] = 0

            dx, dy = enemy['direction']
            new_pos = (ex + dx, ey + dy)

            # Check if valid move
            if self._is_valid_position(new_pos):
                enemy['pos'] = new_pos
                enemy['patrol_steps'] += 1
            else:
                # Hit obstacle, turn randomly
                enemy['direction'] = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                enemy['patrol_steps'] = 0

        elif enemy['behavior'] == 'smart_patrol':
            # Smart patrol: Occasionally move toward agent
            ax, ay = self.agent_pos

            # 20% chance to move toward agent (mild chasing)
            if random.random() < 0.2:
                # Move toward agent
                possible_moves = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_pos = (ex + dx, ey + dy)
                    if self._is_valid_position(new_pos):
                        # Calculate distance to agent
                        dist = abs(new_pos[0] - ax) + abs(new_pos[1] - ay)
                        possible_moves.append((dist, new_pos))

                if possible_moves:
                    # Move closer to agent
                    possible_moves.sort()
                    enemy['pos'] = possible_moves[0][1]
            else:
                # Normal patrol
                if random.random() < 0.15:
                    enemy['direction'] = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

                dx, dy = enemy['direction']
                new_pos = (ex + dx, ey + dy)

                if self._is_valid_position(new_pos):
                    enemy['pos'] = new_pos
                else:
                    enemy['direction'] = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

    def step(self, action):
        """
        Take action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        Returns: (state, reward, done)
        """
        if self.done:
            return self._get_game_state(), 0, True

        # Move agent
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = directions[action]
        new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        reward = 0.0

        # Check wall collision
        if new_pos in self.walls or not self._is_valid_position(new_pos):
            # Can't move into wall, stay in place
            new_pos = self.agent_pos
            reward = -0.1
        else:
            # Valid move
            self.agent_pos = new_pos
            reward = 0.05  # Small reward for moving

            # Update position history (keep last 3 positions)
            self.position_history.append(self.agent_pos)
            if len(self.position_history) > 3:
                self.position_history.pop(0)

        # Check coin collection
        if self.agent_pos in self.coins:
            self.coins.remove(self.agent_pos)
            self.score += 1
            self.total_collected += 1
            reward = 10.0  # Coin reward

            # Victory condition
            if len(self.coins) == 0 or self.total_collected >= self.num_coins:
                self.done = True
                reward += 100.0  # Victory bonus!
                return self._get_game_state(), reward, True

        # Move enemies
        for enemy in self.enemies:
            self._move_enemy(enemy)

        # Check enemy collision
        for enemy in self.enemies:
            if self.agent_pos == enemy['pos']:
                self.lives -= 1
                reward = -20.0  # Enemy collision penalty

                if self.lives <= 0:
                    self.done = True
                    reward -= 50.0  # Death penalty
                    return self._get_game_state(), reward, True
                else:
                    # Respawn at center
                    center = self.world_size // 2
                    self.agent_pos = (center, center)
                    # Reset position history after respawn
                    self.position_history = [self.agent_pos]

        # Step counter
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_game_state(), reward, self.done


if __name__ == '__main__':
    # Test different enemy levels
    print("=" * 70)
    print("LOCAL VIEW COLLECTOR GAME - Progressive Enemy Difficulty")
    print("=" * 70)

    for level in range(5):
        print(f"\nEnemy Level {level}:")
        print("-" * 70)

        game = LocalViewGame(world_size=40, num_coins=20, enemy_level=level)
        state = game.reset()

        print(f"World size: {game.world_size}x{game.world_size}")
        print(f"Coins: {len(game.coins)}")
        print(f"Enemies: {len(game.enemies)}")
        print(f"Walls/Obstacles: {len(game.walls)}")
        print(f"Agent spawn: {game.agent_pos}")

        if game.enemies:
            for i, enemy in enumerate(game.enemies):
                print(f"  Enemy {i}: pos={enemy['pos']}, behavior={enemy['behavior']}")

        # Run a few steps
        print(f"\nRunning 20 steps...")
        for i in range(20):
            action = random.randint(0, 3)
            state, reward, done = game.step(action)
            if done:
                print(f"  Step {i+1}: Game ended! Score: {game.score}")
                break
            if reward > 5:
                print(f"  Step {i+1}: Collected coin at {game.agent_pos}! Score: {game.score}")

        print(f"Final: Agent={game.agent_pos}, Score={game.score}, Lives={game.lives}")

    print("\n" + "=" * 70)
    print("Test complete!")

"""
Simple Dungeon Game with Progressive Enemy Difficulty

Progressive levels:
1. Level 0: Just treasure, no enemies
2. Level 1: 1 patrolling enemy
3. Level 2: 2 patrolling enemies
4. Level 3: 3 patrolling enemies
5. Level 4+: Enemies with smarter patrol patterns
"""
import random


class SimpleDungeonGame:
    """
    Simple Dungeon with progressive enemy difficulty
    """

    def __init__(self, size=20, num_treasures=3, enemy_level=0, max_steps=500, history_length=2):
        """
        Args:
            size: Grid size
            num_treasures: Number of treasures to collect (sparse rewards!)
            enemy_level: 0=no enemies, 1=1 enemy, 2=2 enemies, 3=3 enemies, 4+=smarter
            max_steps: Maximum steps per episode
            history_length: Position history length (2=allow tactical retreat, 3=strict no-back)
        """
        self.size = size
        self.num_treasures = num_treasures
        self.enemy_level = enemy_level
        self.max_steps = max_steps
        self.history_length = history_length  # Shorter for tight corridors!
        self.reset()

    def reset(self):
        """Reset game state"""
        center = self.size // 2

        # Player starts at top-left corner (typical dungeon start)
        self.player_pos = (3, 3)

        # Create dungeon walls FIRST (more complex maze than PacMan)
        self.walls = self._create_dungeon_walls()

        # Place treasures scattered in dungeon (sparse rewards!)
        self.treasures = self._place_treasures()

        # Create patrolling enemies based on level
        self.enemies = self._create_enemies()

        # Game state
        self.score = 0
        self.steps = 0
        self.done = False
        self.lives = 3
        self.total_collected = 0

        # Movement history (prevent immediate backtracking)
        # Store last N positions to prevent oscillation (default: 2 for dungeons)
        # Shorter history allows tactical retreats in tight corridors!
        self.position_history = [self.player_pos]

        return self._get_game_state()

    def _create_dungeon_walls(self):
        """Create dungeon-style maze (more complex than PacMan)"""
        walls = set()

        # Boundary walls
        for i in range(self.size):
            walls.add((0, i))
            walls.add((i, 0))
            walls.add((self.size-1, i))
            walls.add((i, self.size-1))

        # Create maze-like corridors
        # Vertical corridors
        for x in [5, 10, 15]:
            for y in range(2, self.size - 2):
                # Leave gaps for passage
                if y % 5 != 2:
                    walls.add((x, y))

        # Horizontal corridors
        for y in [5, 10, 15]:
            for x in range(2, self.size - 2):
                # Leave gaps for passage
                if x % 5 != 2:
                    walls.add((x, y))

        # Add some rooms (clear spaces)
        rooms = [
            (3, 3, 4, 4),   # Top-left (start room)
            (16, 3, 4, 4),  # Top-right
            (3, 16, 4, 4),  # Bottom-left
            (16, 16, 4, 4), # Bottom-right
        ]

        for rx, ry, rw, rh in rooms:
            for x in range(rx, rx + rw):
                for y in range(ry, ry + rh):
                    walls.discard((x, y))

        return walls

    def _place_treasures(self):
        """Place treasures in dungeon (sparse rewards - only a few!)"""
        treasures = set()

        # Predefined treasure locations (far from start)
        treasure_spots = [
            (self.size - 4, self.size - 4),  # Bottom-right corner
            (self.size - 4, 3),               # Top-right corner
            (3, self.size - 4),               # Bottom-left corner
        ]

        # Shuffle and take num_treasures
        random.shuffle(treasure_spots)
        for i in range(min(self.num_treasures, len(treasure_spots))):
            pos = treasure_spots[i]
            if pos not in self.walls and pos != self.player_pos:
                treasures.add(pos)

        # If we need more treasures, place randomly
        attempts = 0
        max_attempts = 1000
        while len(treasures) < self.num_treasures and attempts < max_attempts:
            x = random.randint(2, self.size - 3)
            y = random.randint(2, self.size - 3)
            pos = (x, y)

            # Valid: not in walls, player start, or existing treasures
            if (pos not in self.walls and
                pos != self.player_pos and
                pos not in treasures):
                treasures.add(pos)

            attempts += 1

        return treasures

    def _create_enemies(self):
        """Create patrolling enemies based on difficulty level"""
        enemies = []

        if self.enemy_level == 0:
            return enemies  # No enemies

        # Number of enemies
        num_enemies = min(self.enemy_level, 3)  # Cap at 3 enemies

        # Enemy spawn positions (in corners/rooms far from player)
        spawn_positions = [
            (self.size - 4, self.size - 4),  # Bottom-right
            (self.size - 4, 4),               # Top-right
            (4, self.size - 4),               # Bottom-left
        ]

        for i in range(num_enemies):
            enemy = {
                'pos': spawn_positions[i],
                'direction': random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)]),
                'behavior': 'patrol' if self.enemy_level < 4 else 'smart_patrol',
                'id': i,
                'patrol_turns': 0  # Track how long moving in one direction
            }
            enemies.append(enemy)

        return enemies

    def _get_game_state(self):
        """Get current game state"""
        # Enemy entities (observer expects dicts with 'pos' key)
        enemy_entities = [{'pos': e['pos'], 'type': 'enemy'} for e in self.enemies]

        # Use position history as "fake tail" to prevent backtracking
        fake_tail = self.position_history[:-1] if len(self.position_history) > 1 else []

        return {
            'agent_pos': self.player_pos,
            'walls': self.walls,
            'rewards': list(self.treasures),
            'entities': enemy_entities,  # Enemies are entities to avoid
            'snake_body': fake_tail,     # Recent positions prevent oscillation
            'grid_size': (self.size, self.size),
            'score': self.score,
            'done': self.done
        }

    def _move_enemy(self, enemy):
        """Move enemy based on behavior"""
        ex, ey = enemy['pos']

        if enemy['behavior'] == 'patrol':
            # Simple patrol: Move in one direction until hitting wall
            # 90% continue, 10% random turn
            if random.random() < 0.1 or enemy['patrol_turns'] > 8:
                enemy['direction'] = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                enemy['patrol_turns'] = 0

            dx, dy = enemy['direction']
            new_pos = (ex + dx, ey + dy)

            # Check if valid move
            if (new_pos not in self.walls and
                1 <= new_pos[0] < self.size - 1 and
                1 <= new_pos[1] < self.size - 1):
                enemy['pos'] = new_pos
                enemy['patrol_turns'] += 1
            else:
                # Hit wall, turn randomly
                enemy['direction'] = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                enemy['patrol_turns'] = 0

        elif enemy['behavior'] == 'smart_patrol':
            # Smart patrol: Try to move toward player occasionally
            px, py = self.player_pos

            # 30% chance to move toward player, 70% random patrol
            if random.random() < 0.3:
                # Move toward player
                possible_moves = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_pos = (ex + dx, ey + dy)
                    if (new_pos not in self.walls and
                        1 <= new_pos[0] < self.size - 1 and
                        1 <= new_pos[1] < self.size - 1):
                        # Calculate distance to player
                        dist = abs(new_pos[0] - px) + abs(new_pos[1] - py)
                        possible_moves.append((dist, new_pos))

                if possible_moves:
                    # Move closer to player
                    possible_moves.sort()
                    enemy['pos'] = possible_moves[0][1]
            else:
                # Normal patrol
                if random.random() < 0.15:
                    enemy['direction'] = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

                dx, dy = enemy['direction']
                new_pos = (ex + dx, ey + dy)

                if (new_pos not in self.walls and
                    1 <= new_pos[0] < self.size - 1 and
                    1 <= new_pos[1] < self.size - 1):
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

        # Move player
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = directions[action]
        new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

        reward = 0.0

        # Check wall collision
        if new_pos in self.walls:
            # Can't move into wall, stay in place
            new_pos = self.player_pos
            reward = -0.1
        else:
            # Valid move
            self.player_pos = new_pos
            reward = 0.05  # Small reward for moving

            # Update position history (keep last N positions to prevent oscillation)
            self.position_history.append(self.player_pos)
            if len(self.position_history) > self.history_length:
                self.position_history.pop(0)

        # Check treasure collection (SPARSE REWARDS!)
        if self.player_pos in self.treasures:
            self.treasures.remove(self.player_pos)
            self.score += 1
            self.total_collected += 1
            reward = 50.0  # Big reward for treasure! (sparse)

            # Victory condition - collected all treasures!
            if len(self.treasures) == 0 or self.total_collected >= self.num_treasures:
                self.done = True
                reward += 200.0  # Victory bonus!
                return self._get_game_state(), reward, True

        # Move enemies
        for enemy in self.enemies:
            self._move_enemy(enemy)

        # Check enemy collision
        for enemy in self.enemies:
            if self.player_pos == enemy['pos']:
                self.lives -= 1
                reward = -30.0  # Enemy collision penalty

                if self.lives <= 0:
                    self.done = True
                    reward -= 100.0  # Death penalty
                    return self._get_game_state(), reward, True
                else:
                    # Respawn at start
                    self.player_pos = (3, 3)
                    # Reset position history after respawn
                    self.position_history = [self.player_pos]

        # Step counter
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_game_state(), reward, self.done


if __name__ == '__main__':
    # Test different enemy levels
    print("=" * 60)
    print("SIMPLE DUNGEON GAME - Progressive Enemy Difficulty")
    print("=" * 60)

    for level in range(5):
        print(f"\nEnemy Level {level}:")
        print("-" * 60)

        game = SimpleDungeonGame(size=20, num_treasures=3, enemy_level=level)
        state = game.reset()

        print(f"Grid size: {game.size}x{game.size}")
        print(f"Treasures: {len(game.treasures)} at {game.treasures}")
        print(f"Enemies: {len(game.enemies)}")

        if game.enemies:
            for i, enemy in enumerate(game.enemies):
                print(f"  Enemy {i}: pos={enemy['pos']}, behavior={enemy['behavior']}")

        # Run a few steps
        print(f"\nRunning 10 steps...")
        for i in range(10):
            action = random.randint(0, 3)
            state, reward, done = game.step(action)
            if done:
                print(f"  Step {i+1}: Game ended! Score: {game.score}")
                break

        print(f"Final state: Player={game.player_pos}, Score={game.score}, Lives={game.lives}")

    print("\n" + "=" * 60)
    print("Test complete!")

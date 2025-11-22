"""
Simple PacMan Game with Progressive Ghost Difficulty

Progressive levels:
1. Level 0: Just pellets, no ghosts
2. Level 1: 1 ghost (moving wall - random walk)
3. Level 2: 2 ghosts (moving walls)
4. Level 3: 3 ghosts (moving walls)
5. Level 4+: Ghosts start chasing (simple AI)
"""
import random


class SimplePacManGame:
    """
    Simple PacMan with progressive ghost difficulty
    """

    def __init__(self, size=20, num_pellets=30, ghost_level=0, max_steps=500):
        """
        Args:
            size: Grid size
            num_pellets: Number of pellets to collect
            ghost_level: 0=no ghosts, 1=1 ghost, 2=2 ghosts, 3=3 ghosts, 4+=chasing
            max_steps: Maximum steps per episode
        """
        self.size = size
        self.num_pellets = num_pellets
        self.ghost_level = ghost_level
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        """Reset game state"""
        center = self.size // 2

        # PacMan starts at center
        self.pacman_pos = (center, center)

        # Create simple maze walls FIRST (so pellets can avoid them)
        self.walls = self._create_walls()

        # Place pellets scattered around (avoiding walls)
        self.pellets = self._place_pellets()

        # Create ghosts based on level
        self.ghosts = self._create_ghosts()

        # Game state
        self.score = 0
        self.steps = 0
        self.done = False
        self.lives = 3
        self.total_collected = 0

        # Movement history (prevent immediate backtracking)
        # Store last 3 positions to prevent oscillation
        self.position_history = [self.pacman_pos]

        return self._get_game_state()

    def _create_walls(self):
        """Create simple maze walls (boundary + some obstacles)"""
        walls = set()

        # Boundary walls
        for i in range(self.size):
            walls.add((0, i))
            walls.add((i, 0))
            walls.add((self.size-1, i))
            walls.add((i, self.size-1))

        # Add a few internal walls for maze feel (optional)
        # Simple cross pattern in quarters
        mid = self.size // 2
        quarter = self.size // 4
        three_quarter = 3 * self.size // 4

        # Horizontal walls
        for i in range(quarter, three_quarter):
            if i != mid:  # Leave center open
                walls.add((i, quarter))
                walls.add((i, three_quarter))

        # Vertical walls
        for i in range(quarter, three_quarter):
            if i != mid:
                walls.add((quarter, i))
                walls.add((three_quarter, i))

        return walls

    def _place_pellets(self):
        """Place pellets randomly (avoiding walls and spawn)"""
        pellets = set()
        center = self.size // 2

        # Safe spawn zone (3x3 around center)
        safe_zone = set()
        for i in range(-1, 2):
            for j in range(-1, 2):
                safe_zone.add((center + i, center + j))

        attempts = 0
        max_attempts = 5000

        while len(pellets) < self.num_pellets and attempts < max_attempts:
            x = random.randint(2, self.size - 3)
            y = random.randint(2, self.size - 3)
            pos = (x, y)

            # Valid position: not in walls, spawn, or existing pellets
            if (pos not in safe_zone and
                pos not in pellets and
                pos not in self.walls):
                pellets.add(pos)

            attempts += 1

        return pellets

    def _create_ghosts(self):
        """Create ghosts based on difficulty level"""
        ghosts = []

        if self.ghost_level == 0:
            return ghosts  # No ghosts

        # Number of ghosts
        num_ghosts = min(self.ghost_level, 3)  # Cap at 3 ghosts

        # Ghost spawn positions (corners)
        spawn_positions = [
            (3, 3),
            (self.size - 4, 3),
            (3, self.size - 4),
        ]

        for i in range(num_ghosts):
            ghost = {
                'pos': spawn_positions[i],
                'direction': random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)]),
                'behavior': 'random' if self.ghost_level < 4 else 'chase',
                'id': i
            }
            ghosts.append(ghost)

        return ghosts

    def _get_game_state(self):
        """Get current game state"""
        # Ghost entities (observer expects dicts with 'pos' key)
        ghost_entities = [{'pos': g['pos'], 'type': 'ghost'} for g in self.ghosts]

        # Use position history as "fake tail" to prevent backtracking
        # This mimics snake's body constraint
        fake_tail = self.position_history[:-1] if len(self.position_history) > 1 else []

        return {
            'agent_pos': self.pacman_pos,
            'walls': self.walls,
            'rewards': list(self.pellets),
            'entities': ghost_entities,  # Ghosts are entities to avoid
            'snake_body': fake_tail,  # Recent positions act like snake body - prevents oscillation
            'grid_size': (self.size, self.size),
            'score': self.score,
            'done': self.done
        }

    def _move_ghost(self, ghost):
        """Move ghost based on behavior"""
        gx, gy = ghost['pos']

        if ghost['behavior'] == 'random':
            # Random walk (moving wall behavior)
            # 80% continue current direction, 20% random turn
            if random.random() < 0.2:
                ghost['direction'] = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

            dx, dy = ghost['direction']
            new_pos = (gx + dx, gy + dy)

            # Check if valid move
            if (new_pos not in self.walls and
                1 <= new_pos[0] < self.size - 1 and
                1 <= new_pos[1] < self.size - 1):
                ghost['pos'] = new_pos
            else:
                # Hit wall, change direction
                ghost['direction'] = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

        elif ghost['behavior'] == 'chase':
            # Simple chase behavior (move toward PacMan)
            px, py = self.pacman_pos

            # Choose direction that reduces distance
            possible_moves = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_pos = (gx + dx, gy + dy)
                if (new_pos not in self.walls and
                    1 <= new_pos[0] < self.size - 1 and
                    1 <= new_pos[1] < self.size - 1):
                    # Calculate distance to PacMan
                    dist = abs(new_pos[0] - px) + abs(new_pos[1] - py)
                    possible_moves.append((dist, new_pos))

            if possible_moves:
                # Move to position closest to PacMan
                possible_moves.sort()
                ghost['pos'] = possible_moves[0][1]

    def step(self, action):
        """
        Take action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        Returns: (state, reward, done)
        """
        if self.done:
            return self._get_game_state(), 0, True

        # Move PacMan
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = directions[action]
        new_pos = (self.pacman_pos[0] + dx, self.pacman_pos[1] + dy)

        reward = 0.0

        # Check wall collision
        if new_pos in self.walls:
            # Can't move into wall, stay in place
            new_pos = self.pacman_pos
            reward = -0.1
        else:
            # Valid move
            self.pacman_pos = new_pos
            reward = 0.05  # Small reward for moving

            # Update position history (keep last 3 positions to prevent oscillation)
            self.position_history.append(self.pacman_pos)
            if len(self.position_history) > 3:
                self.position_history.pop(0)

        # Check pellet collection
        if self.pacman_pos in self.pellets:
            self.pellets.remove(self.pacman_pos)
            self.score += 1
            self.total_collected += 1
            reward = 10.0  # Pellet reward

            # Victory condition
            if len(self.pellets) == 0 or self.total_collected >= self.num_pellets:
                self.done = True
                reward += 100.0  # Victory bonus!
                return self._get_game_state(), reward, True

        # Move ghosts
        for ghost in self.ghosts:
            self._move_ghost(ghost)

        # Check ghost collision
        for ghost in self.ghosts:
            if self.pacman_pos == ghost['pos']:
                self.lives -= 1
                reward = -20.0  # Ghost collision penalty

                if self.lives <= 0:
                    self.done = True
                    reward -= 50.0  # Death penalty
                    return self._get_game_state(), reward, True
                else:
                    # Respawn at center
                    center = self.size // 2
                    self.pacman_pos = (center, center)
                    # Reset position history after respawn
                    self.position_history = [self.pacman_pos]

        # Step counter
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_game_state(), reward, self.done


if __name__ == '__main__':
    # Test different ghost levels
    print("=" * 60)
    print("SIMPLE PACMAN GAME - Progressive Ghost Difficulty")
    print("=" * 60)

    for level in range(5):
        print(f"\nGhost Level {level}:")
        print("-" * 60)

        game = SimplePacManGame(size=20, num_pellets=30, ghost_level=level)
        state = game.reset()

        print(f"Grid size: {game.size}x{game.size}")
        print(f"Pellets: {len(game.pellets)}")
        print(f"Ghosts: {len(game.ghosts)}")

        if game.ghosts:
            for i, ghost in enumerate(game.ghosts):
                print(f"  Ghost {i}: pos={ghost['pos']}, behavior={ghost['behavior']}")

        # Run a few steps
        print(f"\nRunning 10 steps...")
        for i in range(10):
            action = random.randint(0, 3)
            state, reward, done = game.step(action)
            if done:
                print(f"  Step {i+1}: Game ended! Score: {game.score}")
                break

        print(f"Final state: PacMan={game.pacman_pos}, Score={game.score}, Lives={game.lives}")

    print("\n" + "=" * 60)
    print("Test complete!")

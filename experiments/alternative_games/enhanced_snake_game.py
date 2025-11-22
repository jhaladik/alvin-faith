"""
Enhanced Snake Game with Progressive Difficulty

Features:
1. Progressive food count (increases with score)
2. Food disappears after timeout (urgency mechanic)
3. Central obstacles at higher difficulty levels
4. Configurable difficulty progression
"""
import random
from .planning_test_games import SnakeGame


class EnhancedSnakeGame(SnakeGame):
    """
    Enhanced Snake with progressive difficulty features
    """

    def __init__(self, size=20, initial_pellets=3, max_pellets=10,
                 food_timeout=150, obstacle_level=0, max_steps=300,
                 max_total_food=30):
        """
        Args:
            size: Grid size
            initial_pellets: Starting number of food pellets
            max_pellets: Maximum number of pellets (caps progression)
            food_timeout: Steps before food disappears (0 = never)
            obstacle_level: 0=none, 1=few, 2=moderate, 3=many
            max_steps: Maximum steps per episode (prevents farming)
            max_total_food: Victory condition - stop after collecting this many total (0 = no limit)
        """
        self.initial_pellets = initial_pellets
        self.max_pellets = max_pellets
        self.food_timeout = food_timeout
        self.obstacle_level = obstacle_level
        self.food_spawn_times = {}  # Track when each food spawned
        self.central_obstacles = set()
        self.max_total_food = max_total_food

        # Initialize with parent constructor
        super().__init__(size=size, num_pellets=initial_pellets)

        # Override max_steps after parent init
        self.max_steps = max_steps

    def reset(self):
        """Reset with progressive features"""
        result = super().reset()

        # Add central obstacles based on level
        self.central_obstacles = self._create_central_obstacles()

        # Initialize food spawn times
        self.food_spawn_times = {pos: 0 for pos in self.food_positions}

        return result

    def _create_central_obstacles(self):
        """Create scattered obstacles around the grid (avoiding spawn area)"""
        obstacles = set()

        if self.obstacle_level == 0:
            return obstacles  # No obstacles

        center = self.size // 2
        snake_spawn = (center, center)  # Snake always spawns at center

        # Define safe spawn zone (3x3 around center)
        safe_zone = set()
        for i in range(-1, 2):
            for j in range(-1, 2):
                safe_zone.add((center + i, center + j))

        if self.obstacle_level == 1:
            # Level 1: 4 obstacles in corners of safe zone
            positions = [
                (center - 2, center - 2),
                (center + 2, center - 2),
                (center - 2, center + 2),
                (center + 2, center + 2),
            ]
            obstacles.update(positions)

        elif self.obstacle_level == 2:
            # Level 2: 8 obstacles scattered around center
            positions = [
                (center - 3, center - 3),
                (center + 3, center - 3),
                (center - 3, center + 3),
                (center + 3, center + 3),
                (center, center - 3),
                (center, center + 3),
                (center - 3, center),
                (center + 3, center),
            ]
            obstacles.update(positions)

        elif self.obstacle_level >= 3:
            # Level 3+: Scattered obstacles throughout the map
            # Number of obstacles increases with level
            target_obstacles = 10 + (self.obstacle_level - 3) * 5

            attempts = 0
            max_attempts = 5000

            while len(obstacles) < target_obstacles and attempts < max_attempts:
                # Random position avoiding borders (leave 2 tile margin)
                x = random.randint(2, self.size - 3)
                y = random.randint(2, self.size - 3)
                pos = (x, y)

                # Check if valid position
                if (pos not in safe_zone and
                    pos not in obstacles and
                    pos not in self.snake):
                    obstacles.add(pos)

                attempts += 1

        return obstacles

    def _place_all_food(self):
        """Place food avoiding snake, walls, and obstacles"""
        food = set()
        attempts = 0
        max_attempts = 5000

        # Calculate current target based on score (progressive difficulty)
        # Handle case where score hasn't been initialized yet (during first reset)
        current_score = getattr(self, 'score', 0)
        current_target = min(self.initial_pellets + (current_score // 3), self.max_pellets)

        while len(food) < current_target and attempts < max_attempts:
            pos = (random.randint(1, self.size-2), random.randint(1, self.size-2))

            # Check if position is valid (not in snake, food, walls, or obstacles)
            if (pos not in self.snake and
                pos not in food and
                pos not in self.central_obstacles):
                food.add(pos)

            attempts += 1

        return food

    def _get_game_state(self):
        """Get game state including obstacles"""
        state = super()._get_game_state()

        # Add central obstacles to walls
        state['walls'] = state['walls'].union(self.central_obstacles)

        return state

    def step(self, action):
        """Enhanced step with food timeout and progressive difficulty"""
        if self.done:
            return self._get_game_state(), 0, True

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_dir = directions[action]
        if (new_dir[0] + self.direction[0] != 0) or (new_dir[1] + self.direction[1] != 0):
            self.direction = new_dir

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        reward = 0.1  # Movement reward

        # Check wall collision (includes central obstacles)
        if (new_head[0] <= 0 or new_head[0] >= self.size-1 or
            new_head[1] <= 0 or new_head[1] >= self.size-1 or
            new_head in self.central_obstacles):
            self.lives -= 1
            reward = -50.0
            if self.lives <= 0:
                self.done = True
                reward -= 100.0
                return self._get_game_state(), reward, True
            else:
                # Respawn at center
                center = self.size // 2
                self.snake = [(center, center)]
                self.direction = (0, -1)
                return self._get_game_state(), reward, False

        # Self collision
        if new_head in self.snake:
            self.lives -= 1
            reward = -50.0
            if self.lives <= 0:
                self.done = True
                reward -= 100.0
                return self._get_game_state(), reward, True
            else:
                center = self.size // 2
                self.snake = [(center, center)]
                self.direction = (0, -1)
                return self._get_game_state(), reward, False

        self.snake.insert(0, new_head)

        # Food collection
        if new_head in self.food_positions:
            self.food_positions.remove(new_head)
            if new_head in self.food_spawn_times:
                del self.food_spawn_times[new_head]
            self.score += 1
            self.total_collected += 1
            reward = 20.0

            # Check victory condition (max total food collected)
            if self.max_total_food > 0 and self.total_collected >= self.max_total_food:
                self.done = True
                reward += 500.0  # Big victory bonus!
                return self._get_game_state(), reward, True

            # Progressive: Add ONE new food if needed to reach target
            current_score = getattr(self, 'score', 0)
            current_target = min(self.initial_pellets + (current_score // 3), self.max_pellets)

            if len(self.food_positions) < current_target:
                # Add one new food pellet
                attempts = 0
                max_attempts = 1000
                while len(self.food_positions) < current_target and attempts < max_attempts:
                    new_food_pos = (random.randint(1, self.size-2), random.randint(1, self.size-2))

                    if (new_food_pos not in self.snake and
                        new_food_pos not in self.food_positions and
                        new_food_pos not in self.central_obstacles):
                        self.food_positions.add(new_food_pos)
                        self.food_spawn_times[new_food_pos] = self.steps
                        break
                    attempts += 1
        else:
            self.snake.pop()

        # Food timeout mechanic (if enabled)
        if self.food_timeout > 0:
            expired_food = set()
            for food_pos, spawn_time in self.food_spawn_times.items():
                if self.steps - spawn_time > self.food_timeout:
                    expired_food.add(food_pos)

            # Remove expired food
            for expired in expired_food:
                if expired in self.food_positions:
                    self.food_positions.remove(expired)
                if expired in self.food_spawn_times:
                    del self.food_spawn_times[expired]

            # If all food expired, spawn new food
            if len(self.food_positions) == 0:
                self.food_positions = self._place_all_food()
                self.food_spawn_times = {pos: self.steps for pos in self.food_positions}

        # Victory condition (all pellets collected)
        if len(self.food_positions) == 0 and self.food_timeout == 0:
            self.done = True
            reward += 200.0
            return self._get_game_state(), reward, True

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_game_state(), reward, self.done

    def get_difficulty_info(self):
        """Get current difficulty settings"""
        current_score = getattr(self, 'score', 0)
        return {
            'current_food_target': min(self.initial_pellets + (current_score // 3), self.max_pellets),
            'food_timeout': self.food_timeout,
            'obstacle_level': self.obstacle_level,
            'obstacles_count': len(getattr(self, 'central_obstacles', set()))
        }


if __name__ == '__main__':
    # Test enhanced snake
    print("Testing Enhanced Snake Game")
    print("="*60)

    # Test 1: Basic game
    print("\nTest 1: Basic game (no enhancements)")
    game = EnhancedSnakeGame(size=10, initial_pellets=3, obstacle_level=0, food_timeout=0)
    state = game.reset()
    print(f"Food positions: {len(game.food_positions)}")
    print(f"Obstacles: {len(game.central_obstacles)}")

    # Test 2: With obstacles
    print("\nTest 2: With central obstacles (level 2)")
    game = EnhancedSnakeGame(size=10, initial_pellets=3, obstacle_level=2, food_timeout=0)
    state = game.reset()
    print(f"Food positions: {len(game.food_positions)}")
    print(f"Obstacles: {len(game.central_obstacles)}")
    print(f"Obstacle positions: {sorted(list(game.central_obstacles))[:10]}...")

    # Test 3: With food timeout
    print("\nTest 3: With food timeout (150 steps)")
    game = EnhancedSnakeGame(size=10, initial_pellets=3, obstacle_level=0, food_timeout=150)
    state = game.reset()
    print(f"Food spawn times: {game.food_spawn_times}")

    # Simulate 200 steps
    for i in range(200):
        state, reward, done = game.step(1)  # Move down
        if done:
            break

    print(f"After 200 steps: {len(game.food_positions)} food remaining")

    # Test 4: Progressive food
    print("\nTest 4: Progressive food count")
    game = EnhancedSnakeGame(size=15, initial_pellets=3, max_pellets=10, food_timeout=0)
    state = game.reset()

    for score in [0, 3, 6, 9, 12]:
        game.score = score
        food = game._place_all_food()
        target = min(3 + (score // 3), 10)
        print(f"Score {score}: Target food = {target}, Actual = {len(food)}")

    print("\n✓ All tests completed!")

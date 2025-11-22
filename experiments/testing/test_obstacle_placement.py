"""Test obstacle placement in EnhancedSnakeGame"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.enhanced_snake_game import EnhancedSnakeGame

def visualize_grid(game):
    """Visualize the game grid with obstacles and snake"""
    grid = [[' ' for _ in range(game.size)] for _ in range(game.size)]

    # Mark boundaries
    for i in range(game.size):
        grid[0][i] = '#'
        grid[game.size-1][i] = '#'
        grid[i][0] = '#'
        grid[i][game.size-1] = '#'

    # Mark obstacles
    for ox, oy in game.central_obstacles:
        grid[oy][ox] = 'X'

    # Mark food
    for fx, fy in game.food_positions:
        grid[fy][fx] = 'F'

    # Mark snake
    for sx, sy in game.snake[1:]:
        grid[sy][sx] = 's'

    # Mark snake head
    hx, hy = game.snake[0]
    grid[hy][hx] = 'S'

    # Print grid
    for row in grid:
        print(''.join(row))
    print()

def test_obstacle_levels():
    """Test each obstacle level"""
    print("=" * 60)
    print("OBSTACLE PLACEMENT TEST")
    print("=" * 60)

    for level in range(4):
        print(f"\nLevel {level}:")
        print("-" * 60)

        game = EnhancedSnakeGame(
            size=15,
            initial_pellets=3,
            max_pellets=7,
            food_timeout=0,
            obstacle_level=level
        )
        game.reset()

        center = game.size // 2
        snake_pos = game.snake[0]

        print(f"Grid size: {game.size}x{game.size}")
        print(f"Center: {center}")
        print(f"Snake position: {snake_pos}")
        print(f"Obstacles count: {len(game.central_obstacles)}")
        print(f"Obstacle positions: {sorted(list(game.central_obstacles))}")

        # Check if snake can move without hitting obstacle immediately
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        safe_directions = []
        for dx, dy in directions:
            new_pos = (snake_pos[0] + dx, snake_pos[1] + dy)
            if new_pos not in game.central_obstacles:
                safe_directions.append(new_pos)

        print(f"Safe moves from spawn: {len(safe_directions)}/4")
        print(f"Safe positions: {safe_directions}")

        # Visualize
        print("\nVisualization (S=snake, X=obstacle, F=food, #=wall):")
        visualize_grid(game)

        if len(safe_directions) < 4:
            print("[WARNING] Snake is trapped at spawn!")
        else:
            print("[OK] Snake has room to move")

if __name__ == '__main__':
    test_obstacle_levels()

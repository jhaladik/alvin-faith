"""
Test to verify wall boundaries are correct in both training and testing
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "core"))

from src.core.planning_test_games import SnakeGame

def test_wall_boundaries():
    """Test wall positions and collision detection"""
    print("="*70)
    print("WALL BOUNDARY TEST")
    print("="*70)

    game = SnakeGame(size=20, num_pellets=10)
    game.reset()

    print(f"\nGrid size: {game.size} x {game.size}")
    print(f"Expected playable area: (1, 1) to ({game.size-2}, {game.size-2})")
    print(f"Expected walls at: x=0, x={game.size-1}, y=0, y={game.size-1}")

    # Check actual walls
    state = game._get_game_state()
    walls = state['walls']

    print(f"\nTotal walls: {len(walls)}")

    # Check corners
    corners = [
        (0, 0), (0, game.size-1),
        (game.size-1, 0), (game.size-1, game.size-1)
    ]
    print("\nCorner walls:")
    for corner in corners:
        is_wall = corner in walls
        print(f"  {corner}: {'[WALL]' if is_wall else '[NOT WALL]'}")

    # Check edges
    print("\nEdge positions:")
    print(f"  Top edge (y=0): (0,0) to ({game.size-1},0)")
    print(f"  Bottom edge (y={game.size-1}): (0,{game.size-1}) to ({game.size-1},{game.size-1})")
    print(f"  Left edge (x=0): (0,0) to (0,{game.size-1})")
    print(f"  Right edge (x={game.size-1}): ({game.size-1},0) to ({game.size-1},{game.size-1})")

    # Check collision logic
    print("\n" + "="*70)
    print("COLLISION DETECTION TEST")
    print("="*70)

    test_positions = [
        ((0, 10), "Left wall"),
        ((19, 10), "Right wall"),
        ((10, 0), "Top wall"),
        ((10, 19), "Bottom wall"),
        ((1, 10), "Just inside left"),
        ((18, 10), "Just inside right"),
        ((10, 1), "Just inside top"),
        ((10, 18), "Just inside bottom"),
    ]

    for pos, description in test_positions:
        # Reset and place snake at position
        game.reset()
        game.snake = [pos]

        # Try to move right (for left/right walls)
        if "left" in description.lower() or "right" in description.lower():
            test_action = 3  # RIGHT
            next_pos = (pos[0] + 1, pos[1])
        else:  # Try to move down (for top/bottom walls)
            test_action = 1  # DOWN
            next_pos = (pos[0], pos[1] + 1)

        # Check if next position would collide
        would_collide = (next_pos[0] <= 0 or next_pos[0] >= game.size-1 or
                        next_pos[1] <= 0 or next_pos[1] >= game.size-1)

        is_wall = next_pos in walls

        print(f"\n{description}: {pos}")
        print(f"  Next position: {next_pos}")
        print(f"  Is wall in set: {is_wall}")
        print(f"  Collision check: {would_collide}")
        print(f"  Match: {'[OK]' if (is_wall == would_collide) else '[MISMATCH!]'}")


def test_actual_collision():
    """Test actual collision by moving snake"""
    print("\n" + "="*70)
    print("ACTUAL COLLISION TEST")
    print("="*70)

    game = SnakeGame(size=20, num_pellets=10)

    # Test 1: Move from near-wall toward wall
    print("\nTest 1: Snake at (2, 10), moving LEFT toward wall at (0, 10)")
    game.reset()
    game.snake = [(2, 10)]
    game.direction = (-1, 0)  # LEFT

    print(f"  Initial position: {game.snake[0]}")
    print(f"  Initial lives: {game.lives}")

    # Move once (to x=1)
    state1, reward1, done1 = game.step(2)  # LEFT
    print(f"  After move 1: position={game.snake[0]}, lives={game.lives}, reward={reward1:.1f}")

    # Move again (to x=0 - wall!)
    state2, reward2, done2 = game.step(2)  # LEFT
    print(f"  After move 2: position={game.snake[0]}, lives={game.lives}, reward={reward2:.1f}")

    if reward2 < -10:
        print(f"  [OK] Collision detected! Penalty: {reward2:.1f}")
    else:
        print(f"  [ERROR] No collision detected? Reward: {reward2:.1f}")

    # Test 2: Move from center repeatedly until collision
    print("\nTest 2: Snake starting at center, moving RIGHT until collision")
    game.reset()
    game.snake = [(10, 10)]
    game.direction = (1, 0)  # RIGHT

    moves = 0
    max_moves = 20
    while not game.done and moves < max_moves:
        prev_lives = game.lives
        prev_pos = game.snake[0]
        state, reward, done = game.step(3)  # RIGHT
        moves += 1

        if prev_lives > game.lives:
            print(f"  Move {moves}: COLLISION at {prev_pos} -> reward={reward:.1f}, lives={game.lives}")
            break

        if moves % 5 == 0:
            print(f"  Move {moves}: position={game.snake[0]}, lives={game.lives}")


if __name__ == '__main__':
    test_wall_boundaries()
    test_actual_collision()

"""
Test script to verify snake body detection fix
"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "core"))

from src.core.planning_test_games import SnakeGame
from src.context_aware_agent import ContextAwareDQN, add_context_to_observation
from src.core.expanded_temporal_observer import ExpandedTemporalObserver


def test_snake_body_detection():
    """Test if the observer now correctly detects snake body segments"""
    print("="*70)
    print("TEST: Snake Body Detection After Fix")
    print("="*70)

    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15, verbose=False)
    game = SnakeGame(size=20, num_pellets=10)

    # Create a snake with body segments
    game.reset()
    # Make snake have a longer body: head at (10,10), body extending down
    game.snake = [(10, 10), (10, 11), (10, 12), (10, 13)]
    game.direction = (0, -1)  # Moving UP

    state = game._get_game_state()

    print(f"\nSnake configuration:")
    print(f"  Head: {game.snake[0]}")
    print(f"  Body: {game.snake[1:]}")
    print(f"  Direction: {game.direction} (moving UP)")
    print(f"  Snake body in state: {state.get('snake_body', 'NOT PRESENT')}")

    # Get observation
    observer.reset()
    obs = observer.observe(state)

    # Extract wall distances from observation
    # First 16*3 features are: [reward_dist, entity_dist, wall_dist] per ray
    wall_dists = obs[2::3][:16]  # Every 3rd element starting from index 2

    print(f"\nWall distances (including snake body):")
    print(f"  Min: {wall_dists.min():.3f}, Max: {wall_dists.max():.3f}")
    print(f"  Closest obstacle: ray {wall_dists.argmin()}, distance {wall_dists.min():.3f} (normalized)")
    print(f"  Actual distance: {wall_dists.min() * 15:.1f} tiles")

    # Ray 4 should point DOWN (toward body)
    # With 16 rays: ray 0 = RIGHT (1,0), ray 4 = DOWN (0,1), ray 8 = LEFT (-1,0), ray 12 = UP (0,-1)
    ray_down_idx = 4
    print(f"\nRay pointing DOWN (toward body, ray {ray_down_idx}):")
    print(f"  Distance: {wall_dists[ray_down_idx]:.3f} (normalized)")
    print(f"  Actual: {wall_dists[ray_down_idx] * 15:.1f} tiles")
    print(f"  Expected: ~1 tile (body starts at distance 1)")

    if wall_dists[ray_down_idx] < 0.2:  # Less than 3 tiles
        print(f"  [OK] Snake body detected!")
    else:
        print(f"  [WARNING] Snake body NOT detected! Distance too large.")

    return wall_dists[ray_down_idx] < 0.2


def test_longer_snake_avoidance():
    """Test agent behavior with longer snake"""
    print("\n" + "="*70)
    print("TEST: Agent With Longer Snake Body")
    print("="*70)

    checkpoint_path = "checkpoints/snake_focused_20251121_031557_policy.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    agent = ContextAwareDQN(obs_dim=183, action_dim=4)
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15, verbose=False)
    game = SnakeGame(size=20, num_pellets=10)

    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    # Test: Snake facing its own body
    print("\nScenario: Snake about to collide with its own body")

    game.reset()
    # Snake with head at (10,10), body going down, now turning to face right and about to turn down into body
    game.snake = [(10, 10), (9, 10), (8, 10), (7, 10), (7, 11), (7, 12), (8, 12), (9, 12), (10, 12)]
    game.direction = (1, 0)  # Facing RIGHT

    state = game._get_game_state()
    print(f"  Snake head: {game.snake[0]}")
    print(f"  Snake body length: {len(game.snake)}")
    print(f"  Direction: RIGHT")
    print(f"  Body segment at (10,12) - 2 tiles DOWN from head")

    observer.reset()
    obs = observer.observe(state)
    context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    obs_with_context = add_context_to_observation(obs, context)

    with torch.no_grad():
        action = agent.get_action(obs_with_context, epsilon=0.0)
        obs_tensor = torch.FloatTensor(obs_with_context).unsqueeze(0)
        q_values = agent.get_combined_q(obs_tensor)[0].numpy()

    print(f"\n  Agent chooses: {action_names[action]}")
    print(f"  Q-values:")
    for i, (name, q) in enumerate(zip(action_names, q_values)):
        marker = " <-- CHOSEN" if i == action else ""
        danger = ""
        if name == "DOWN":
            danger = " [WOULD HIT BODY!]"
        print(f"    {name}: {q:.3f}{marker}{danger}")

    # Verify action avoids body
    if action == 1:  # DOWN
        print(f"\n  [WARNING] Agent chose DOWN - would collide with body!")
        return False
    else:
        print(f"\n  [OK] Agent avoided body collision")
        return True


if __name__ == '__main__':
    success1 = test_snake_body_detection()
    success2 = test_longer_snake_avoidance()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Snake body detection: {'[OK]' if success1 else '[FAILED]'}")
    print(f"Agent avoidance: {'[OK]' if success2 else '[FAILED]'}")

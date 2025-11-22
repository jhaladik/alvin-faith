"""
Test script to debug wall detection and avoidance in snake game
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


def test_wall_raycasting():
    """Test if the observer correctly detects walls"""
    print("="*70)
    print("TEST 1: Wall Detection in Raycasting")
    print("="*70)

    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15, verbose=True)
    game = SnakeGame(size=20, num_pellets=10)

    # Test positions near walls
    test_positions = [
        ((10, 10), "Center - far from walls"),
        ((2, 10), "Near left wall"),
        ((18, 10), "Near right wall"),
        ((10, 2), "Near top wall"),
        ((10, 18), "Near bottom wall"),
        ((2, 2), "Near corner"),
    ]

    for pos, description in test_positions:
        print(f"\n{description}: Position {pos}")

        # Create a game state at this position
        game.reset()
        game.snake = [pos]
        state = game._get_game_state()

        # Get observation
        observer.reset()
        obs = observer.observe(state)

        # Extract wall distances from observation
        # First 16*3 features are: [reward_dist, entity_dist, wall_dist] per ray
        wall_dists = obs[2::3][:16]  # Every 3rd element starting from index 2

        print(f"  Wall distances (16 rays):")
        print(f"    Min: {wall_dists.min():.3f}, Max: {wall_dists.max():.3f}, Mean: {wall_dists.mean():.3f}")
        print(f"    Closest wall: ray {wall_dists.argmin()}, distance {wall_dists.min():.3f} (normalized)")
        print(f"    Actual distance: {wall_dists.min() * 15:.1f} tiles")

        # Check if walls are properly detected
        min_wall_dist = wall_dists.min() * 15  # Denormalize
        if "near" in description.lower():
            if min_wall_dist > 5:
                print(f"  [WARNING] Near wall but min distance is {min_wall_dist:.1f} tiles!")
            else:
                print(f"  [OK] Wall detected correctly")


def test_agent_decisions_near_walls():
    """Test what actions the agent chooses when near walls"""
    print("\n" + "="*70)
    print("TEST 2: Agent Decisions Near Walls")
    print("="*70)

    # Load checkpoint
    checkpoint_path = "checkpoints/snake_focused_20251121_031557_policy.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    agent = ContextAwareDQN(obs_dim=183, action_dim=4)
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15, verbose=False)
    game = SnakeGame(size=20, num_pellets=10)

    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    # Test scenarios where snake is near walls and heading toward them
    scenarios = [
        ((2, 10), (1, 0), "Near LEFT wall, facing LEFT"),
        ((18, 10), (-1, 0), "Near RIGHT wall, facing RIGHT"),
        ((10, 2), (0, 1), "Near TOP wall, facing UP"),
        ((10, 18), (0, -1), "Near BOTTOM wall, facing DOWN"),
    ]

    for pos, direction, description in scenarios:
        print(f"\n{description}")
        print(f"  Position: {pos}, Direction: {direction}")

        # Setup game state
        game.reset()
        game.snake = [pos]
        game.direction = direction
        state = game._get_game_state()

        # Get observation
        observer.reset()
        obs = observer.observe(state)

        # Add context (snake mode)
        context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        obs_with_context = add_context_to_observation(obs, context)

        # Get agent's action (epsilon=0 for deterministic)
        with torch.no_grad():
            action = agent.get_action(obs_with_context, epsilon=0.0)

        print(f"  Agent chooses: {action_names[action]} (action {action})")

        # Check if this would hit a wall
        next_pos = (pos[0] + direction[0], pos[1] + direction[1])
        if next_pos[0] <= 0 or next_pos[0] >= 19 or next_pos[1] <= 0 or next_pos[1] >= 19:
            print(f"  [WARNING] Continuing in same direction would HIT WALL!")

            # Show Q-values for all actions
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs_with_context).unsqueeze(0)
                q_values = agent.get_combined_q(obs_tensor)[0].numpy()

            print(f"  Q-values for all actions:")
            for i, (name, q) in enumerate(zip(action_names, q_values)):
                marker = " <-- CHOSEN" if i == action else ""
                print(f"    {name}: {q:.3f}{marker}")


def run_episode_and_count_collisions():
    """Run a full episode and count wall collisions"""
    print("\n" + "="*70)
    print("TEST 3: Full Episode - Count Wall Collisions")
    print("="*70)

    # Load checkpoint
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

    # Run 5 episodes
    for ep in range(5):
        print(f"\nEpisode {ep+1}:")

        state = game.reset()
        observer.reset()

        wall_collisions = 0
        self_collisions = 0
        food_collected = 0
        steps = 0

        while not state['done'] and steps < 1000:
            obs = observer.observe(state)
            context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            obs_with_context = add_context_to_observation(obs, context)

            with torch.no_grad():
                action = agent.get_action(obs_with_context, epsilon=0.0)

            prev_lives = game.lives
            prev_score = game.score
            state, reward, done = game.step(action)
            steps += 1

            # Check what happened
            if prev_lives > game.lives:
                # Lost a life - was it wall or self collision?
                head = game.snake[0]
                if head[0] <= 0 or head[0] >= 19 or head[1] <= 0 or head[1] >= 19:
                    wall_collisions += 1
                else:
                    self_collisions += 1

            if game.score > prev_score:
                food_collected += 1

        print(f"  Steps: {steps}")
        print(f"  Food collected: {food_collected}")
        print(f"  Wall collisions: {wall_collisions}")
        print(f"  Self collisions: {self_collisions}")
        print(f"  Final score: {game.score}")
        print(f"  Lives remaining: {game.lives}")


if __name__ == '__main__':
    test_wall_raycasting()
    test_agent_decisions_near_walls()
    run_episode_and_count_collisions()

"""
Test agent behavior when approaching walls
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


def test_agent_near_walls():
    """Test what agent does when near walls"""
    print("="*70)
    print("AGENT BEHAVIOR NEAR WALLS")
    print("="*70)

    # Load checkpoint
    checkpoint_path = "checkpoints/snake_focused_20251121_031557_policy.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading: {checkpoint_path}\n")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    agent = ContextAwareDQN(obs_dim=183, action_dim=4)
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15, verbose=False)
    game = SnakeGame(size=20, num_pellets=10)

    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    # Test: Put snake near right wall and see what it does
    print("Test: Snake at (17, 10), wall at (19, 10)")
    print("Should avoid moving RIGHT\n")

    game.reset()
    game.snake = [(17, 10)]
    game.direction = (1, 0)  # Facing RIGHT

    for step in range(5):
        state = game._get_game_state()

        print(f"\nStep {step+1}:")
        print(f"  Position: {game.snake[0]}")
        print(f"  Lives: {game.lives}")

        # Get observation
        observer.reset() if step == 0 else None
        obs = observer.observe(state)

        # Check wall distances
        wall_dists = obs[2::3][:16]  # Extract wall distances
        print(f"  Wall distances - Min: {wall_dists.min():.3f} ({wall_dists.min()*15:.1f} tiles)")

        # Ray 0 points RIGHT
        wall_right = wall_dists[0]
        print(f"  Wall RIGHT (ray 0): {wall_right:.3f} ({wall_right*15:.1f} tiles)")

        # Get agent action
        context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        obs_with_context = add_context_to_observation(obs, context)

        with torch.no_grad():
            action = agent.get_action(obs_with_context, epsilon=0.0)
            obs_tensor = torch.FloatTensor(obs_with_context).unsqueeze(0)
            q_values = agent.get_combined_q(obs_tensor)[0].numpy()

        print(f"  Q-values: UP={q_values[0]:.1f}, DOWN={q_values[1]:.1f}, LEFT={q_values[2]:.1f}, RIGHT={q_values[3]:.1f}")
        print(f"  Agent chooses: {action_names[action]}")

        # Step
        next_state, reward, done = game.step(action)

        if reward < -10:
            print(f"  >>> COLLISION! Reward: {reward:.1f}")
            break
        else:
            print(f"  Moved successfully, reward: {reward:.1f}")

        if done or game.snake[0][0] >= 19:
            break


def test_full_episode_wall_awareness():
    """Run full episode and track wall collision scenarios"""
    print("\n" + "="*70)
    print("FULL EPISODE WALL AWARENESS TEST")
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

    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    state = game.reset()
    observer.reset()

    wall_warnings = []
    steps = 0
    max_steps = 100

    print("\nRunning episode...")
    print("Tracking situations where wall is close but agent moves toward it\n")

    while not state['done'] and steps < max_steps:
        obs = observer.observe(state)
        wall_dists = obs[2::3][:16]

        context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        obs_with_context = add_context_to_observation(obs, context)

        with torch.no_grad():
            action = agent.get_action(obs_with_context, epsilon=0.0)

        # Check if moving toward close wall
        pos = game.snake[0]
        min_wall_dist = wall_dists.min() * 15

        # Ray 0=RIGHT, 4=DOWN, 8=LEFT, 12=UP
        ray_dirs = {0: "RIGHT", 4: "DOWN", 8: "LEFT", 12: "UP"}
        action_to_ray = {3: 0, 1: 4, 2: 8, 0: 12}  # Action -> Ray index

        if action in action_to_ray:
            ray_idx = action_to_ray[action]
            wall_in_direction = wall_dists[ray_idx] * 15

            if wall_in_direction < 3.0:  # Wall within 3 tiles
                warning = {
                    'step': steps,
                    'pos': pos,
                    'action': action_names[action],
                    'wall_dist': wall_in_direction,
                    'min_wall': min_wall_dist
                }
                wall_warnings.append(warning)
                print(f"Step {steps}: pos={pos}, moving {action_names[action]}, wall {wall_in_direction:.1f} tiles away")

        prev_lives = game.lives
        state, reward, done = game.step(action)
        steps += 1

        if prev_lives > game.lives:
            print(f"  >>> COLLISION! Lost life at step {steps}")

    print(f"\nEpisode complete:")
    print(f"  Steps: {steps}")
    print(f"  Wall warnings (moved toward wall <3 tiles): {len(wall_warnings)}")
    print(f"  Final lives: {game.lives}")
    print(f"  Score: {game.score}")


if __name__ == '__main__':
    test_agent_near_walls()
    test_full_episode_wall_awareness()

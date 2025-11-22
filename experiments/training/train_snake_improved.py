"""
Improved Snake Training - Fixes Wall Collision, Wandering, AND Circling

Improvements over train_snake_focused.py:
1. Stronger wall penalty (-100 vs -50)
2. Slower epsilon decay (70% vs 50% of episodes)
3. Curriculum learning (grid: 10→15→20, progressive features)
4. Danger-zone reward shaping (weaker: -0.5 when <1.2 tiles)
5. NET progress reward (anti-circling: rewards real progress over 5 steps)
6. Stronger collection bonus (100 + combo*20 vs 50 + combo*10)
7. Proximity bonus (+5 when adjacent to food)
8. Prioritized replay for collisions (2x priority weight)
9. Progressive difficulty features:
   - Food count increases with score (3→7→12 max)
   - Food disappears after timeout (urgency mechanic, 200→150 steps)
   - Central obstacles at higher levels (0→1→2: none→small cross→large cross)
10. Stagnation penalty (-0.5 if no collection for 30 steps)

Fixes:
- Wall collisions: Stronger penalty + curriculum learning
- Wandering: Gentler danger penalty + stagnation penalty
- Circling food: Net progress tracking prevents oscillation
"""
import torch
import torch.optim as optim
import numpy as np
import random
from datetime import datetime
from collections import deque

from context_aware_agent import ContextAwareDQN, add_context_to_observation
from core.expanded_temporal_observer import ExpandedTemporalObserver
from core.context_aware_world_model import ContextAwareWorldModel
from core.enhanced_snake_game import EnhancedSnakeGame
from train_context_aware_advanced import PrioritizedReplayBuffer


class ImprovedSnakeRewards:
    """
    Anti-circling reward system with net progress tracking
    """
    def __init__(self):
        self.combo_count = 0
        self.steps_alive = 0
        self.food_dist_history = deque(maxlen=5)  # Track progress over 5 steps
        self.steps_since_collection = 0

    def reset(self):
        self.combo_count = 0
        self.steps_alive = 0
        self.food_dist_history.clear()
        self.steps_since_collection = 0

    def calculate(self, base_reward, died, min_wall_dist, nearest_food_dist=None, prev_nearest_food_dist=None):
        """
        Anti-circling rewards with net progress:
        - Pellet: 100 base + combo*20 (MUCH STRONGER to overcome circling)
        - Survival: 0.1 per step
        - Death: -100
        - Danger zone: -0.5 if <1.2 tiles (WEAKER to allow collection)
        - Net progress: Rewards real progress over 5 steps (ANTI-CIRCLING)
        - Proximity: +5 when adjacent to food (encourages final approach)
        - Stagnation: -0.5 if no collection for 30 steps
        """
        total = 0.0

        # 1. Pellet collection (MUCH STRONGER: 100 + combo*20)
        if base_reward >= 10:
            pellet_reward = 100.0 + (self.combo_count * 20.0)
            total += pellet_reward
            self.combo_count += 1
            self.steps_since_collection = 0  # Reset stagnation counter

        # 2. Survival
        self.steps_alive += 1
        self.steps_since_collection += 1
        total += 0.1

        # 3. Death penalty
        if died:
            total -= 100.0

        # 4. Danger zone (WEAKER: -0.5 when <1.2 tiles)
        if min_wall_dist < 1.2:
            danger_penalty = -0.5 * (1.0 - min_wall_dist / 1.2)
            total += danger_penalty

        # 5. NET progress reward (ANTI-CIRCLING)
        if nearest_food_dist is not None:
            self.food_dist_history.append(nearest_food_dist)

            if len(self.food_dist_history) >= 5:
                # Only reward NET progress over 5 steps
                initial_dist = self.food_dist_history[0]
                current_dist = nearest_food_dist
                net_progress = initial_dist - current_dist

                if net_progress > 0:
                    total += net_progress * 2.0  # Reward real progress
                elif net_progress < -2:
                    total -= 1.0  # Penalty for moving significantly away

            # 6. Proximity bonus (encourages final approach)
            if nearest_food_dist == 1:
                total += 5.0  # Big bonus for being adjacent to food!

        # 7. Stagnation penalty (prevents wandering)
        if self.steps_since_collection > 30:
            total -= 0.5

        return total


def train_snake_improved(episodes=500, batch_size=64, learning_rate=1e-4):
    """Train with improved strategy for wall avoidance"""

    print("=" * 70)
    print("IMPROVED SNAKE TRAINING - Anti-Circling Edition")
    print("=" * 70)
    print(f"Episodes: {episodes}")
    print()
    print("KEY IMPROVEMENTS:")
    print("  1. Stronger wall penalty: -100 (vs -50)")
    print("  2. Slower epsilon decay: 70% of episodes (vs 50%)")
    print("  3. Curriculum learning: 10→15→20 grid size")
    print("  4. Danger-zone shaping: -0.5 when <1.2 tiles (WEAKER)")
    print("  5. NET progress reward: Prevents circling (5-step window)")
    print("  6. Stronger collection: 100 + combo*20 (vs 50 + combo*10)")
    print("  7. Proximity bonus: +5 when adjacent to food")
    print("  8. Stagnation penalty: -0.5 after 30 steps without collection")
    print("  9. Progressive difficulty:")
    print("     - Food count increases with score")
    print("     - Food timeout (200→150 steps)")
    print("     - Central obstacles (none→small→large cross)")
    print("  10. Prioritized replay: 2x weight for collisions")
    print()

    # Initialize
    device = torch.device('cpu')
    policy_net = ContextAwareDQN(obs_dim=183, action_dim=4).to(device)
    target_net = ContextAwareDQN(obs_dim=183, action_dim=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = PrioritizedReplayBuffer(capacity=50000)

    # Observer
    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)

    # Reward system
    reward_system = ImprovedSnakeRewards()

    # Training parameters
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = episodes * 0.7  # SLOWER decay (350 episodes vs 250)

    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    episode_wall_collisions = []

    for episode in range(1, episodes + 1):
        # CURRICULUM LEARNING: Progressive difficulty with obstacles and timeouts
        if episode < 150:
            game_size = 10  # Small grid (walls very close)
            curriculum_phase = "SMALL"
            # Early: Simple (3 food start, no obstacles, no timeout, victory at 15)
            game = EnhancedSnakeGame(
                size=game_size,
                initial_pellets=3,
                max_pellets=7,
                food_timeout=0,  # No timeout yet
                obstacle_level=0,  # No obstacles
                max_steps=200,  # Shorter episodes
                max_total_food=15  # Victory after 15 pellets
            )
        elif episode < 350:
            game_size = 15  # Medium grid
            curriculum_phase = "MEDIUM"
            # Mid: Moderate (5 food start, few obstacles, victory at 25)
            game = EnhancedSnakeGame(
                size=game_size,
                initial_pellets=5,
                max_pellets=10,
                food_timeout=200,  # Long timeout
                obstacle_level=1,  # Small cross in center
                max_steps=300,  # Medium episodes
                max_total_food=25  # Victory after 25 pellets
            )
        else:
            game_size = 20  # Full grid
            curriculum_phase = "FULL"
            # Late: Hard (7 food start, more obstacles, victory at 40)
            game = EnhancedSnakeGame(
                size=game_size,
                initial_pellets=7,
                max_pellets=12,
                food_timeout=150,  # Shorter timeout
                obstacle_level=2,  # Larger cross
                max_steps=400,  # Longer episodes for larger grid
                max_total_food=40  # Victory after 40 pellets
            )

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon_start - (episode / epsilon_decay))

        # Reset
        state = game.reset()
        observer.reset()
        reward_system.reset()

        episode_reward = 0.0
        episode_length = 0
        wall_collisions_this_episode = 0
        done = False
        prev_nearest_food_dist = None  # Track food distance for approach reward

        while not done:
            # Observe
            obs = observer.observe(state)
            context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)  # Snake context
            obs_with_context = add_context_to_observation(obs, context)

            # Calculate min wall distance for danger-zone shaping
            wall_dists = obs[2::3][:16]  # Extract wall distances from observation
            min_wall_dist_normalized = wall_dists.min()
            min_wall_dist_tiles = min_wall_dist_normalized * 15  # Denormalize

            # Calculate nearest food distance for approach reward
            head_pos = game.snake[0]
            food_positions = list(game.food_positions)
            if food_positions:
                nearest_food_dist = min([abs(head_pos[0] - f[0]) + abs(head_pos[1] - f[1])
                                        for f in food_positions])
            else:
                nearest_food_dist = None

            # Select action
            action = policy_net.get_action(obs_with_context, epsilon=epsilon)

            # Step
            prev_lives = game.lives
            next_state, base_reward, done = game.step(action)

            # Check if wall collision occurred
            died = (prev_lives > game.lives)
            if died and base_reward < -40:  # Wall collision (not self-collision)
                head = game.snake[0] if game.snake else (0, 0)
                if (head[0] <= 0 or head[0] >= game_size-1 or
                    head[1] <= 0 or head[1] >= game_size-1):
                    wall_collisions_this_episode += 1

            # Enhanced reward with danger-zone shaping AND approach-food reward
            enhanced_reward = reward_system.calculate(
                base_reward, died, min_wall_dist_tiles,
                nearest_food_dist, prev_nearest_food_dist
            )

            # Update prev_nearest_food_dist for next iteration
            prev_nearest_food_dist = nearest_food_dist

            # Next observation
            next_obs = observer.observe(next_state)
            next_obs_with_context = add_context_to_observation(next_obs, context)

            # Store transition
            transition = (obs_with_context, action, enhanced_reward, next_obs_with_context, done)
            replay_buffer.add(transition)

            episode_reward += enhanced_reward
            episode_length += 1
            state = next_state

            # Train
            if len(replay_buffer) >= batch_size:
                # Sample batch
                transitions, indices, weights = replay_buffer.sample(batch_size)

                if transitions is None:
                    continue

                states = torch.FloatTensor(np.stack([t[0] for t in transitions])).to(device)
                actions = torch.LongTensor([t[1] for t in transitions]).to(device)
                rewards = torch.FloatTensor([t[2] for t in transitions]).to(device)
                next_states = torch.FloatTensor(np.stack([t[3] for t in transitions])).to(device)
                dones = torch.FloatTensor([t[4] for t in transitions]).to(device)

                # Q-learning update
                q_values = policy_net.get_combined_q(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_q_values = target_net.get_combined_q(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)

                # Loss
                td_errors = torch.abs(q_values - target_q_values)
                loss = torch.mean(td_errors ** 2)

                # PRIORITIZED REPLAY: Higher priority for collisions
                priorities = td_errors.detach().cpu().numpy()
                for i, reward in enumerate(rewards):
                    if reward < -50:  # Collision experience
                        priorities[i] *= 2.0  # Double the priority!

                replay_buffer.update_priorities(indices, priorities)

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        # Track stats
        episode_rewards.append(episode_reward)
        episode_scores.append(game.score)
        episode_lengths.append(episode_length)
        episode_wall_collisions.append(wall_collisions_this_episode)

        # Update target network
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Log progress
        if episode % 50 == 0 or episode == episodes:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_score = np.mean(episode_scores[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            avg_wall_collisions = np.mean(episode_wall_collisions[-50:])

            # Get difficulty info from game
            difficulty = game.get_difficulty_info()

            print(f"\nEpisode {episode}/{episodes} [{curriculum_phase} grid: {game_size}x{game_size}]")
            print(f"  Difficulty: Food={difficulty['current_food_target']}, " +
                  f"Timeout={difficulty['food_timeout']}, " +
                  f"Obstacles={difficulty['obstacles_count']}")
            print(f"  Avg Reward (100): {avg_reward:.2f}")
            print(f"  Avg Score (100): {avg_score:.2f}")
            print(f"  Avg Length (100): {avg_length:.1f}")
            print(f"  Avg Wall Collisions (50): {avg_wall_collisions:.2f}")
            print(f"  Epsilon: {epsilon:.3f}")
            print(f"  Buffer: {len(replay_buffer)}")

            # Recent performance
            recent_scores = episode_scores[-10:]
            recent_wall_collisions = episode_wall_collisions[-10:]
            print(f"  Recent 10 scores: {recent_scores}")
            print(f"  Recent 10 wall collisions: {recent_wall_collisions}")
            print(f"  Best recent: {max(recent_scores)}")

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save to repository root checkpoints folder
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)  # Go up one level from src/
    checkpoints_dir = os.path.join(repo_root, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    save_path = os.path.join(checkpoints_dir, f"snake_improved_{timestamp}_policy.pth")

    torch.save({
        'episode': episodes,
        'policy_net': policy_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_scores': episode_scores,
        'episode_lengths': episode_lengths,
        'episode_wall_collisions': episode_wall_collisions,
        'improvements': {
            'wall_penalty': -100,
            'epsilon_decay': 0.7,
            'curriculum': True,
            'danger_shaping': True,
            'prioritized_collisions': True,
            'net_progress_tracking': True,
            'stronger_collection_bonus': True,
            'progressive_food': True,
            'food_timeout': True,
            'central_obstacles': True,
            'stagnation_penalty': True
        }
    }, save_path)

    print(f"\n✓ Model saved: {save_path}")
    print()
    print("=" * 70)
    print("TRAINING COMPLETE - Anti-Circling Edition")
    print("=" * 70)
    print(f"Final Avg Score (100): {np.mean(episode_scores[-100:]):.2f}")
    print(f"Final Avg Reward (100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final Avg Length (100): {np.mean(episode_lengths[-100:]):.1f}")
    print(f"Final Avg Wall Collisions (50): {np.mean(episode_wall_collisions[-50:]):.2f}")
    print(f"Best Score: {max(episode_scores)}")
    print()
    print("EXPECTED IMPROVEMENTS:")
    print("  - Wall collisions: <0.5 per episode")
    print("  - Episode length: 50-80 steps (vs previous 150-170)")
    print("  - Score: 8-12 pellets collected efficiently")
    print("  - No circling around food (net progress tracking works)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train_snake_improved(
        episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

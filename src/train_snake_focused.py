"""
Snake-Only Focused Training

Prove the agent can master Snake when properly trained:
- 100% Snake episodes (no context mixing)
- Balanced reward structure (collection vs survival)
- 500 episodes should be sufficient
- Planning + Faith enabled for comparison
"""
import torch
import torch.optim as optim
import numpy as np
import random
from datetime import datetime

from context_aware_agent import ContextAwareDQN, add_context_to_observation
from core.expanded_temporal_observer import ExpandedTemporalObserver
from core.context_aware_world_model import ContextAwareWorldModel
from core.planning_test_games import SnakeGame
from train_context_aware_advanced import PrioritizedReplayBuffer

# Simplified reward system - balanced for Snake
class SnakeFocusedRewards:
    """Reward system optimized for Snake collection"""
    def __init__(self):
        self.combo_count = 0
        self.steps_alive = 0

    def reset(self):
        self.combo_count = 0
        self.steps_alive = 0

    def calculate(self, base_reward, died):
        """
        Balanced rewards:
        - Pellet: 50 base + combo bonus (emphasize collection!)
        - Survival: 0.1 per step (modest)
        - Death: -100 (fixed penalty)
        """
        total = 0.0

        # Pellet collection (base_reward = 10 from game)
        if base_reward >= 10:
            pellet_reward = 50.0 + (self.combo_count * 10.0)  # Big rewards for collection!
            total += pellet_reward
            self.combo_count += 1

        # Survival (modest)
        self.steps_alive += 1
        total += 0.1

        # Death penalty (fixed)
        if died:
            total -= 100.0

        return total


def train_snake_focused(episodes=500, batch_size=64, learning_rate=1e-4):
    """Train exclusively on Snake to prove capability"""

    print("=" * 70)
    print("SNAKE-ONLY FOCUSED TRAINING")
    print("=" * 70)
    print(f"Episodes: {episodes}")
    print(f"Game: Snake (100% focus)")
    print(f"Reward: Balanced (50 per pellet + combo, 0.1 per step)")
    print(f"Planning: DISABLED (reactive only for baseline)")
    print(f"Faith: DISABLED (reactive only for baseline)")
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

    # Game
    game = SnakeGame(size=20, num_pellets=10)

    # Reward system
    reward_system = SnakeFocusedRewards()

    # Training loop
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = episodes * 0.5

    episode_rewards = []
    episode_scores = []
    episode_lengths = []

    for episode in range(1, episodes + 1):
        # Epsilon decay
        epsilon = max(epsilon_end, epsilon_start - (episode / epsilon_decay))

        # Reset
        state = game.reset()
        observer.reset()
        reward_system.reset()

        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # Observe
            obs = observer.observe(state)
            context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)  # Snake context
            obs_with_context = add_context_to_observation(obs, context)

            # Select action (pure Q-learning, no planning/faith)
            action = policy_net.get_action(obs_with_context, epsilon=epsilon)

            # Step
            next_state, base_reward, done = game.step(action)

            # Enhanced reward
            enhanced_reward = reward_system.calculate(base_reward, died=done)

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

                # Use np.stack() to properly stack arrays with same shape
                states = torch.FloatTensor(np.stack([t[0] for t in transitions])).to(device)
                actions = torch.LongTensor([t[1] for t in transitions]).to(device)
                rewards = torch.FloatTensor([t[2] for t in transitions]).to(device)
                next_states = torch.FloatTensor(np.stack([t[3] for t in transitions])).to(device)
                dones = torch.FloatTensor([t[4] for t in transitions]).to(device)

                # Q-learning update (use get_combined_q to handle multi-head architecture)
                q_values = policy_net.get_combined_q(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_q_values = target_net.get_combined_q(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)

                # Loss
                td_errors = torch.abs(q_values - target_q_values)
                loss = torch.mean(td_errors ** 2)

                # Update priorities
                replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        # Track stats
        episode_rewards.append(episode_reward)
        episode_scores.append(game.score)
        episode_lengths.append(episode_length)

        # Update target network
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Log progress
        if episode % 50 == 0 or episode == episodes:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_score = np.mean(episode_scores[-100:])
            avg_length = np.mean(episode_lengths[-100:])

            print(f"\nEpisode {episode}/{episodes}")
            print(f"  Avg Reward (100): {avg_reward:.2f}")
            print(f"  Avg Score (100): {avg_score:.2f}")
            print(f"  Avg Length (100): {avg_length:.1f}")
            print(f"  Epsilon: {epsilon:.3f}")
            print(f"  Buffer: {len(replay_buffer)}")

            # Recent performance
            recent_scores = episode_scores[-10:]
            print(f"  Recent 10 scores: {recent_scores}")
            print(f"  Best recent: {max(recent_scores)}")

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"checkpoints/snake_focused_{timestamp}_policy.pth"

    torch.save({
        'episode': episodes,
        'policy_net': policy_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_scores': episode_scores,
        'episode_lengths': episode_lengths,
    }, save_path)

    print(f"\n✓ Model saved: {save_path}")
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final Avg Score (100): {np.mean(episode_scores[-100:]):.2f}")
    print(f"Final Avg Reward (100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Best Score: {max(episode_scores)}")
    print()
    print("Expected: 8-10 avg score if agent learned properly")
    print("If still poor, reward system may need further adjustment")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train_snake_focused(
        episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

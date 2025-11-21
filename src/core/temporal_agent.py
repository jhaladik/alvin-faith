"""
Temporal-Aware Hierarchical DQN Agent
Uses temporal flow observations for informed decision making.

Philosophy: First UNDERSTAND (perception), then DECIDE (action).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class TemporalHierarchicalDQN(nn.Module):
    """
    Hierarchical DQN with temporal understanding.

    Architecture:
    1. Perception Network: Understands current + temporal context
    2. Q-Heads: Make decisions based on understood state
    """

    def __init__(self, obs_dim=92, action_dim=4, hidden_dim=128):
        """
        Args:
            obs_dim: 92 = 48 current + 44 delta features
            action_dim: 4 actions (UP, DOWN, LEFT, RIGHT)
            hidden_dim: Hidden layer size (smaller than before - cleaner signal)
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # === PERCEPTION NETWORK ===
        # Understand the situation (current state + temporal context)
        # Smaller network because observation is already rich and structured
        self.perception_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # === HIERARCHICAL Q-HEADS ===
        # Each head makes decisions about its priority

        # SURVIVE: "Will this action kill me?"
        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # AVOID: "How dangerous is this action?"
        self.avoidance_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # POSITION: "Does this action improve my tactical position?"
        self.positioning_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # COLLECT: "Does this action move me toward rewards?"
        self.collection_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Priority weights (adjustable at inference)
        # BALANCED WEIGHTS - Collection is now EQUALLY IMPORTANT as avoidance!
        self.priority_weights = {
            'survive': 8.0,    # Emergency situations (death imminent)
            'avoid': 4.0,      # Keep distance from entities (but don't obsess)
            'position': 2.0,   # Tactical positioning (walls, escape routes)
            'collect': 8.0     # ACTIVELY SEEK REWARDS (equal to survive!)
        }

    def forward(self, x):
        """
        Forward pass: perception -> decision

        Returns dict of Q-values for each priority head.
        """
        # Understand the situation
        state_understanding = self.perception_net(x)

        # Make decisions
        return {
            'survive': self.survival_head(state_understanding),
            'avoid': self.avoidance_head(state_understanding),
            'position': self.positioning_head(state_understanding),
            'collect': self.collection_head(state_understanding)
        }

    def get_combined_q(self, x):
        """Get weighted combination of all Q-heads"""
        q_dict = self.forward(x)

        combined = (
            self.priority_weights['survive'] * q_dict['survive'] +
            self.priority_weights['avoid'] * q_dict['avoid'] +
            self.priority_weights['position'] * q_dict['position'] +
            self.priority_weights['collect'] * q_dict['collect']
        )

        return combined

    def get_action(self, state, epsilon=0.0):
        """Select action using epsilon-greedy on combined Q-values"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.get_combined_q(state_tensor)
            return q_values.argmax(dim=1).item()

    def set_inference_weights(self, survive=100.0, avoid=10.0, position=1.0, collect=0.1):
        """Adjust priority weights at inference time"""
        self.priority_weights = {
            'survive': survive,
            'avoid': avoid,
            'position': position,
            'collect': collect
        }
        print(f"Updated inference weights: {self.priority_weights}")


class TemporalTrainer:
    """
    Trainer for temporal-aware hierarchical agent.

    Key difference: Uses temporal deltas for understanding, not frame stacking.
    """

    def __init__(self, env, device='cpu'):
        """
        Args:
            env: Environment with temporal observer
            device: Training device
        """
        self.env = env
        self.device = device

        # Networks
        obs_dim = env.obs_dim
        self.policy_net = TemporalHierarchicalDQN(obs_dim=obs_dim).to(device)
        self.target_net = TemporalHierarchicalDQN(obs_dim=obs_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)

        # Replay buffer
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 64

        # Training parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.target_update_freq = 500
        self.steps = 0

        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []

    def compute_hierarchical_rewards(self, old_state, action, new_state, info):
        """
        Compute separate reward signals for each Q-head.

        Uses temporal features for richer reward shaping.
        """
        # Extract features from new state
        # Current features: 0-47, Delta features: 48-91

        # Survival reward (NOT constant - based on immediate danger!)
        nearest_entity_dist = new_state[40]  # dist_norm (0-1, lower=closer)
        if info.get('died', False):
            r_survive = -10.0
        else:
            # Immediate danger check: entity within certain range = DANGER
            if nearest_entity_dist < 0.2:  # Very close = danger!
                r_survive = -1.0
            elif nearest_entity_dist < 0.4:  # Close = warning
                r_survive = 0.0
            else:
                r_survive = 0.2  # Safe = good

        # Avoidance reward (based on entity distance - continuous signal)
        # Entity info starts at index 38: [rel_x, rel_y, dist_norm, danger, ...]
        r_avoid = nearest_entity_dist - 0.5  # Reward for being far from entities

        # Positioning reward (based on escape routes AND wall proximity)
        # Topology at 32-37: [corridor, junction, dead_end, openness, escapes, density]
        escape_routes = new_state[36]  # escape_routes normalized
        wall_density = new_state[37]   # wall density (higher = more walls nearby)

        # Wall distances are in ray data: indices 2, 5, 8, 11, 14, 17, 20, 23
        # (every 3rd value starting from index 2)
        min_wall_dist = min(new_state[2], new_state[5], new_state[8], new_state[11],
                           new_state[14], new_state[17], new_state[20], new_state[23])

        # Reward for: good escape routes, far from walls, low wall density
        r_position = escape_routes - 0.5 + (min_wall_dist - 0.3) - wall_density * 0.5

        # Collection reward (using EXPLICIT reward direction!)
        # Reward direction at 46-47: [dx, dy] pointing to nearest reward
        reward_dx = new_state[46]
        reward_dy = new_state[47]

        # Distance to nearest reward (lower = closer = better)
        # Use inverse of reward direction magnitude as proximity signal
        reward_dist = np.sqrt(reward_dx**2 + reward_dy**2) if (reward_dx != 0 or reward_dy != 0) else 1.0
        reward_proximity = 1.0 - min(reward_dist, 1.0)  # 1.0 = very close, 0.0 = far

        # Did we collect?
        if info.get('collected_reward', False):
            r_collect = 10.0
        else:
            # CONTINUOUS reward for being close to reward (like r_avoid for entities)
            # This makes collection equally learnable as avoidance!
            r_collect = reward_proximity - 0.3  # Range: -0.3 to +0.7

            # Bonus for moving TOWARD reward (progress rate from deltas)
            progress_rate = new_state[90] if len(new_state) > 90 else 0.0
            r_collect += progress_rate * 2.0  # Add progress signal

        return {
            'survive': r_survive,
            'avoid': r_avoid,
            'position': r_position,
            'collect': r_collect
        }

    def store_transition(self, state, action, rewards, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, rewards, next_state, done))

    def train_step(self):
        """Train on a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards_list, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current Q-values
        current_q = self.policy_net(states)
        next_q = self.target_net(next_states)

        # Compute loss for each head
        total_loss = 0

        for head_name in ['survive', 'avoid', 'position', 'collect']:
            # Extract rewards for this head
            head_rewards = torch.FloatTensor([r[head_name] for r in rewards_list]).to(self.device)

            # Current Q for taken actions
            q_current = current_q[head_name].gather(1, actions.unsqueeze(1)).squeeze()

            # Target Q (max Q of next state)
            q_next_max = next_q[head_name].max(dim=1)[0]
            q_target = head_rewards + self.gamma * q_next_max * (1 - dones)

            # MSE loss
            loss = nn.MSELoss()(q_current, q_target)
            total_loss += loss

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return total_loss.item()

    def train(self, num_episodes=1000, log_every=100, save_path='temporal_agent.pth'):
        """Train the agent"""
        best_reward = float('-inf')

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0

            done = False
            while not done:
                # Select action
                action = self.policy_net.get_action(state, self.epsilon)

                # Take step
                next_state, env_reward, done, info = self.env.step(action)

                # Compute hierarchical rewards
                rewards = self.compute_hierarchical_rewards(state, action, next_state, info)

                # Store transition
                self.store_transition(state, action, rewards, next_state, done)

                # Train
                self.train_step()

                episode_reward += env_reward
                episode_length += 1
                state = next_state

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Track
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Log
            if (episode + 1) % log_every == 0:
                avg_reward = np.mean(self.episode_rewards[-log_every:])
                avg_length = np.mean(self.episode_lengths[-log_every:])
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.1f}")
                print(f"  Avg Length: {avg_length:.0f}")
                print(f"  Epsilon: {self.epsilon:.4f}")
                print()

                # Save best
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save(save_path.replace('.pth', '_best.pth'))

        # Final save
        self.save(save_path)
        return self.episode_rewards, self.episode_lengths

    def save(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_weights': self.policy_net.priority_weights
        }
        torch.save(checkpoint, path)
        print(f"Saved: {path}")

    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        print(f"Loaded: {path}")


def demo():
    """Quick demo of temporal agent architecture"""
    print("=" * 60)
    print("TEMPORAL HIERARCHICAL AGENT")
    print("First understand, then decide")
    print("=" * 60)
    print()

    agent = TemporalHierarchicalDQN(obs_dim=92)
    print(f"Total parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print()

    # Compare to old agent
    print("Architecture comparison:")
    print("  Old (frame-stacking): 232 input -> 256 hidden -> Q-heads")
    print("  New (temporal flow):   92 input -> 128 hidden -> Q-heads")
    print()
    print("Key improvements:")
    print("  1. Smaller input (92 vs 232) - all features meaningful")
    print("  2. Explicit reward direction in observation")
    print("  3. Temporal deltas encoded directly")
    print("  4. No noise padding - zeros mean 'no history'")
    print()

    # Test with sample input
    sample_obs = np.random.randn(92).astype(np.float32)
    q_values = agent.get_combined_q(torch.FloatTensor(sample_obs).unsqueeze(0))
    print(f"Sample Q-values: {q_values.detach().numpy()[0]}")
    print(f"Best action: {['UP', 'DOWN', 'LEFT', 'RIGHT'][q_values.argmax().item()]}")


if __name__ == '__main__':
    demo()

"""
World Model for Prediction-Based AI Agent
Predicts future states to enable planning and lookahead.

Philosophy: "Imagine before you act"
- Current state + action -> Predicted next state
- Predict reward and done signals
- Plan multiple steps ahead in imagination
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class WorldModelNetwork(nn.Module):
    """
    Learns to predict next state given current state and action.

    This is the "imagination" module - agent can simulate futures.
    """

    def __init__(self, state_dim=92, action_dim=4, hidden_dim=256):
        """
        Args:
            state_dim: Dimension of temporal observation (92 standard, 180 expanded)
            action_dim: Number of possible actions (4)
            hidden_dim: Hidden layer size (256 for expanded, 128 for standard)
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Input: state + one-hot action
        input_dim = state_dim + action_dim

        # State prediction network
        # Predicts: next_state
        self.state_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  # Output: predicted next state
        )

        # Reward prediction network
        # Predicts: scalar reward
        self.reward_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Done prediction network
        # Predicts: probability of episode ending
        self.done_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability 0-1
        )

    def forward(self, state, action):
        """
        Predict next state, reward, and done.

        Args:
            state: Current state tensor (batch_size, state_dim)
            action: Action indices (batch_size,) or one-hot (batch_size, action_dim)

        Returns:
            predicted_next_state: (batch_size, state_dim)
            predicted_reward: (batch_size, 1)
            predicted_done: (batch_size, 1)
        """
        # Convert action to one-hot if needed
        if action.dim() == 1:
            action_onehot = torch.zeros(action.size(0), self.action_dim)
            action_onehot.scatter_(1, action.unsqueeze(1), 1.0)
        else:
            action_onehot = action

        # Concatenate state and action
        x = torch.cat([state, action_onehot], dim=1)

        # Predict
        next_state = self.state_predictor(x)
        reward = self.reward_predictor(x)
        done = self.done_predictor(x)

        return next_state, reward, done

    def imagine_trajectory(self, initial_state, action_sequence):
        """
        Imagine a sequence of future states.

        Args:
            initial_state: Starting state (state_dim,)
            action_sequence: List of actions to take

        Returns:
            List of (state, reward, done) predictions
        """
        trajectory = []
        current_state = initial_state.unsqueeze(0) if initial_state.dim() == 1 else initial_state

        with torch.no_grad():
            for action in action_sequence:
                action_tensor = torch.tensor([action])
                next_state, reward, done = self.forward(current_state, action_tensor)
                trajectory.append({
                    'state': next_state.squeeze().numpy(),
                    'reward': reward.item(),
                    'done': done.item()
                })
                current_state = next_state

                # Stop if episode predicted to end
                if done.item() > 0.5:
                    break

        return trajectory


class WorldModelTrainer:
    """
    Trains world model from experience.

    Learns: (s, a) -> (s', r, done)
    """

    def __init__(self, state_dim=92, action_dim=4, device='cpu'):
        self.device = device
        self.model = WorldModelNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Experience buffer
        self.buffer = deque(maxlen=100000)
        self.batch_size = 128

        # Training stats
        self.train_losses = []

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience for training"""
        self.buffer.append((state, action, reward, next_state, float(done)))

    def train_step(self):
        """Train world model on batch of experiences"""
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Predict
        pred_next_states, pred_rewards, pred_dones = self.model(states, actions)

        # Compute losses
        state_loss = nn.MSELoss()(pred_next_states, next_states)
        reward_loss = nn.MSELoss()(pred_rewards, rewards)
        done_loss = nn.BCELoss()(pred_dones, dones)

        # Total loss (weighted)
        total_loss = state_loss + 0.1 * reward_loss + 0.1 * done_loss

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.train_losses.append(total_loss.item())

        return {
            'total': total_loss.item(),
            'state': state_loss.item(),
            'reward': reward_loss.item(),
            'done': done_loss.item()
        }

    def save(self, path):
        """Save world model"""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_losses': self.train_losses
        }, path)
        print(f"World model saved: {path}")

    def load(self, path):
        """Load world model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_losses = checkpoint.get('train_losses', [])
        print(f"World model loaded: {path}")


class PlanningAgent:
    """
    Agent that uses world model for planning.

    Instead of just reacting, it imagines futures and picks best path.
    """

    def __init__(self, policy_net, world_model, planning_horizon=5):
        """
        Args:
            policy_net: Trained temporal hierarchical DQN
            world_model: Trained world model for prediction
            planning_horizon: How many steps to look ahead
        """
        self.policy = policy_net
        self.world_model = world_model
        self.planning_horizon = planning_horizon
        self.action_dim = 4

    def plan_action(self, current_state, num_simulations=20):
        """
        Use imagination to plan best action.

        1. Imagine multiple action sequences
        2. Predict outcomes using world model
        3. Evaluate trajectories using Q-values
        4. Pick action that leads to best future
        """
        current_state_tensor = torch.FloatTensor(current_state)

        best_action = 0
        best_value = float('-inf')

        # Try each first action
        for first_action in range(self.action_dim):
            total_value = 0

            # Run multiple simulations from this first action
            for _ in range(num_simulations // self.action_dim):
                trajectory_value = self._simulate_trajectory(
                    current_state_tensor, first_action
                )
                total_value += trajectory_value

            avg_value = total_value / (num_simulations // self.action_dim)

            if avg_value > best_value:
                best_value = avg_value
                best_action = first_action

        return best_action, best_value

    def _simulate_trajectory(self, state, first_action):
        """
        Simulate one trajectory into the future.

        Returns: Total discounted value of trajectory
        """
        gamma = 0.99
        total_value = 0

        current_state = state.unsqueeze(0)
        action = first_action

        # First step
        with torch.no_grad():
            action_tensor = torch.tensor([action])
            next_state, pred_reward, pred_done = self.world_model(current_state, action_tensor)
            total_value += pred_reward.item()

            if pred_done.item() > 0.5:
                return total_value

            current_state = next_state

        # Continue planning
        for step in range(1, self.planning_horizon):
            # Use policy to select next action in imagination
            with torch.no_grad():
                q_values = self.policy.get_combined_q(current_state)
                action = q_values.argmax(dim=1).item()

                action_tensor = torch.tensor([action])
                next_state, pred_reward, pred_done = self.world_model(current_state, action_tensor)

                # Discounted reward
                total_value += (gamma ** step) * pred_reward.item()

                if pred_done.item() > 0.5:
                    break

                current_state = next_state

        # Add terminal value estimate (Q-value of final state)
        with torch.no_grad():
            final_q = self.policy.get_combined_q(current_state)
            terminal_value = final_q.max().item()
            total_value += (gamma ** self.planning_horizon) * terminal_value

        return total_value


def demo():
    """Demonstrate world model prediction"""
    print("=" * 60)
    print("WORLD MODEL - IMAGINATION & PREDICTION")
    print("Agent can simulate future before acting")
    print("=" * 60)
    print()

    # Create world model
    world_model = WorldModelNetwork(state_dim=92, action_dim=4)
    print(f"World model parameters: {sum(p.numel() for p in world_model.parameters()):,}")
    print()

    # Simulate state and action
    current_state = torch.randn(1, 92)  # Random current state
    action = torch.tensor([2])  # LEFT action

    # Predict future
    next_state, reward, done = world_model(current_state, action)

    print("Prediction example:")
    print(f"  Current state shape: {current_state.shape}")
    print(f"  Action: LEFT (2)")
    print(f"  Predicted next state shape: {next_state.shape}")
    print(f"  Predicted reward: {reward.item():.3f}")
    print(f"  Predicted done prob: {done.item():.3f}")
    print()

    # Imagine a trajectory
    print("Imagining 5-step trajectory: [RIGHT, UP, UP, LEFT, RIGHT]")
    trajectory = world_model.imagine_trajectory(current_state.squeeze(), [3, 0, 0, 2, 3])
    for i, step in enumerate(trajectory):
        print(f"  Step {i+1}: reward={step['reward']:.3f}, done_prob={step['done']:.3f}")
    print()

    print("KEY INSIGHT:")
    print("The agent can now IMAGINE futures before acting!")
    print("It can ask: 'What if I go LEFT? What if I go RIGHT?'")
    print("Then choose the action with best predicted outcome.")
    print()
    print("This enables:")
    print("  - Avoiding traps (imagine getting cornered)")
    print("  - Path planning (imagine route to reward)")
    print("  - Risk assessment (imagine entity approaching)")


if __name__ == '__main__':
    demo()

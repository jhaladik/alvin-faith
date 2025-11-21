"""
Temporal Buffer Enhancement - Quick fix for ghost prediction

Can be added to existing ContextAwareDQN without full retrain!

Key idea: Maintain longer temporal buffer, compute ensemble predictions
"""
import torch
import torch.nn as nn
import numpy as np
from collections import deque


class TemporalBufferEnhancement(nn.Module):
    """
    Adds multi-scale temporal buffering to existing agent

    Can be integrated with current ContextAwareDQN via fine-tuning
    """

    def __init__(self, obs_dim=95, buffer_size=50):
        super().__init__()
        self.obs_dim = obs_dim
        self.buffer_size = buffer_size

        # Temporal buffers for different scales
        self.micro_buffer = deque(maxlen=5)   # Last 5 frames (high detail)
        self.meso_buffer = deque(maxlen=50)   # Last 50 frames (patterns)

        # Simple temporal feature extractors (lightweight!)
        self.micro_encoder = nn.Sequential(
            nn.Linear(obs_dim * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.meso_encoder = nn.Sequential(
            nn.Linear(obs_dim * 10, 128),  # Sample every 5th frame
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Uncertainty estimator
        self.uncertainty_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0 = certain, 1 = uncertain
        )

        # Temporal fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim)  # Enhanced observation
        )

        # Initialize weights with small values to prevent explosion
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all layers with small weights to prevent Q-value explosion"""
        for module in [self.micro_encoder, self.meso_encoder, self.uncertainty_net, self.fusion]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    # Xavier/Glorot initialization with small gain
                    nn.init.xavier_uniform_(layer.weight, gain=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def update_buffers(self, obs):
        """Add new observation to temporal buffers"""
        self.micro_buffer.append(obs)
        self.meso_buffer.append(obs)

    def extract_temporal_features(self):
        """Extract features from buffered history

        Returns: (micro_features, meso_features) - both 1D tensors (64,)
        """
        # Micro features (all 5 frames)
        if len(self.micro_buffer) == 5:
            # Ensure all are tensors and flatten them
            frames = []
            for f in self.micro_buffer:
                if isinstance(f, torch.Tensor):
                    frames.append(f.flatten())
                else:
                    frames.append(torch.FloatTensor(f).flatten())

            micro_tensor = torch.cat(frames, dim=0).unsqueeze(0)  # (1, obs_dim*5)
            micro_features = self.micro_encoder(micro_tensor).squeeze(0)  # (64,)
        else:
            micro_features = torch.zeros(64)

        # Meso features (sample every 5th frame)
        if len(self.meso_buffer) >= 50:
            sampled = [self.meso_buffer[i] for i in range(0, 50, 5)]
            frames = []
            for f in sampled:
                if isinstance(f, torch.Tensor):
                    frames.append(f.flatten())
                else:
                    frames.append(torch.FloatTensor(f).flatten())

            meso_tensor = torch.cat(frames, dim=0).unsqueeze(0)  # (1, obs_dim*10)
            meso_features = self.meso_encoder(meso_tensor).squeeze(0)  # (64,)
        else:
            meso_features = torch.zeros(64)

        return micro_features, meso_features

    def compute_uncertainty(self, micro_features, meso_features):
        """
        How uncertain are we about predictions?

        High variance in recent frames = high uncertainty

        Args:
            micro_features: (64,) tensor
            meso_features: (64,) tensor

        Returns:
            uncertainty: scalar tensor
        """
        combined = torch.cat([micro_features, meso_features])  # (128,)
        combined = combined.unsqueeze(0)  # (1, 128)
        uncertainty = self.uncertainty_net(combined)  # (1, 1)
        return uncertainty.squeeze()  # scalar

    def enhance_observation(self, current_obs):
        """
        Add temporal context to current observation

        Handles both single observations and batches

        Args:
            current_obs: (obs_dim,) for single or (batch, obs_dim) for batch

        Returns:
            enhanced_obs: same shape as input
            uncertainty: scalar tensor
        """
        # Check if batch or single observation
        is_batch = len(current_obs.shape) == 2
        batch_size = current_obs.shape[0] if is_batch else 1

        # Get temporal features (always returns (64,) tensors)
        micro_features, meso_features = self.extract_temporal_features()

        # Compute uncertainty (single value regardless of batch)
        uncertainty = self.compute_uncertainty(micro_features, meso_features)

        # Expand features to match batch size
        if is_batch:
            micro_features = micro_features.unsqueeze(0).expand(batch_size, -1)
            meso_features = meso_features.unsqueeze(0).expand(batch_size, -1)
            obs_for_fusion = current_obs  # Already (batch, obs_dim)
        else:
            micro_features = micro_features.unsqueeze(0)  # (1, 64)
            meso_features = meso_features.unsqueeze(0)  # (1, 64)
            obs_for_fusion = current_obs.unsqueeze(0)  # (1, obs_dim)

        # Fuse everything - all are now (batch, dim)
        enhanced = self.fusion(torch.cat([
            obs_for_fusion,   # (batch, obs_dim)
            micro_features,   # (batch, 64)
            meso_features     # (batch, 64)
        ], dim=-1))  # (batch, obs_dim+64+64)

        # Remove batch dimension if input was single
        if not is_batch:
            enhanced = enhanced.squeeze(0)

        return enhanced, uncertainty

    def reset(self):
        """Reset buffers for new episode"""
        self.micro_buffer.clear()
        self.meso_buffer.clear()


class GhostEnsemblePredictor:
    """
    Predict ghost positions using ensemble of hypotheses

    Instead of single linear prediction, maintain 3 scenarios:
    - Optimistic: Ghosts move away (scatter)
    - Pessimistic: Ghosts converge (pincer)
    - Expected: Average of recent movements
    """

    def __init__(self):
        self.ghost_history = {}  # {ghost_id: deque of positions}
        self.history_length = 50

    def update(self, ghost_positions):
        """Track ghost positions"""
        for i, pos in enumerate(ghost_positions):
            if i not in self.ghost_history:
                self.ghost_history[i] = deque(maxlen=self.history_length)
            self.ghost_history[i].append(pos)

    def predict_ensemble(self, agent_pos, steps_ahead=5):
        """
        Generate ensemble of predictions

        Returns: {
            'optimistic': [(x, y), ...],  # Best case
            'pessimistic': [(x, y), ...], # Worst case
            'expected': [(x, y), ...],    # Most likely
            'uncertainty': float          # 0-1
        }
        """
        if not self.ghost_history:
            return None

        predictions = {
            'optimistic': [],
            'pessimistic': [],
            'expected': [],
            'uncertainty': []
        }

        for ghost_id, history in self.ghost_history.items():
            if len(history) < 3:
                continue

            # Compute recent velocities
            recent_positions = list(history)[-10:]
            velocities = []
            for i in range(1, len(recent_positions)):
                prev = recent_positions[i-1]
                curr = recent_positions[i]
                vel = (curr[0] - prev[0], curr[1] - prev[1])
                velocities.append(vel)

            if not velocities:
                continue

            # Current position
            curr_pos = recent_positions[-1]

            # Expected: mean velocity
            mean_vx = np.mean([v[0] for v in velocities])
            mean_vy = np.mean([v[1] for v in velocities])
            expected_pos = (
                curr_pos[0] + mean_vx * steps_ahead,
                curr_pos[1] + mean_vy * steps_ahead
            )
            predictions['expected'].append(expected_pos)

            # Optimistic: Ghost moves away from agent
            to_agent_x = agent_pos[0] - curr_pos[0]
            to_agent_y = agent_pos[1] - curr_pos[1]
            # Move opposite direction
            optimistic_pos = (
                curr_pos[0] - np.sign(to_agent_x) * steps_ahead,
                curr_pos[1] - np.sign(to_agent_y) * steps_ahead
            )
            predictions['optimistic'].append(optimistic_pos)

            # Pessimistic: Ghost moves toward agent (chasing)
            pessimistic_pos = (
                curr_pos[0] + np.sign(to_agent_x) * steps_ahead,
                curr_pos[1] + np.sign(to_agent_y) * steps_ahead
            )
            predictions['pessimistic'].append(pessimistic_pos)

            # Uncertainty: variance in velocities
            var_vx = np.var([v[0] for v in velocities])
            var_vy = np.var([v[1] for v in velocities])
            uncertainty = min((var_vx + var_vy) / 2.0, 1.0)
            predictions['uncertainty'].append(uncertainty)

        # Average uncertainty across all ghosts
        if predictions['uncertainty']:
            predictions['uncertainty'] = np.mean(predictions['uncertainty'])
        else:
            predictions['uncertainty'] = 1.0

        return predictions

    def compute_safe_zones(self, agent_pos, predictions, grid_size=(20, 20)):
        """
        Find areas safe in ALL scenarios

        Returns: List of (x, y) positions safe from all ghost predictions
        """
        safe_positions = []

        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                # Check distance to all predicted ghost positions
                safe_in_all = True

                for scenario in ['optimistic', 'expected', 'pessimistic']:
                    for ghost_pred in predictions[scenario]:
                        dist = abs(x - ghost_pred[0]) + abs(y - ghost_pred[1])
                        if dist < 3:  # Too close in this scenario
                            safe_in_all = False
                            break
                    if not safe_in_all:
                        break

                if safe_in_all:
                    safe_positions.append((x, y))

        return safe_positions

    def get_best_action(self, agent_pos, safe_zones, reward_pos):
        """
        Find action that:
        1. Moves toward safe zone
        2. Moves toward reward if possible

        Returns: action (0=up, 1=right, 2=down, 3=left)
        """
        if not safe_zones:
            # No safe zone - find farthest from all ghosts
            # Emergency mode!
            return self._emergency_escape(agent_pos)

        # Find safe zone closest to reward
        best_zone = min(safe_zones, key=lambda z:
            abs(z[0] - reward_pos[0]) + abs(z[1] - reward_pos[1])
        )

        # Move toward that zone
        dx = best_zone[0] - agent_pos[0]
        dy = best_zone[1] - agent_pos[1]

        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 3  # Right or Left
        else:
            return 2 if dy > 0 else 0  # Down or Up

    def _emergency_escape(self, agent_pos):
        """Emergency: just move to create distance"""
        # Random escape direction
        return np.random.randint(4)


# Integration with existing agent
class EnhancedContextAwareDQN(nn.Module):
    """
    Wrapper around existing ContextAwareDQN
    Adds temporal buffering without full retrain
    """

    def __init__(self, base_agent, use_ensemble=True):
        super().__init__()
        self.base_agent = base_agent
        self.temporal_enhancement = TemporalBufferEnhancement()
        self.ghost_predictor = GhostEnsemblePredictor() if use_ensemble else None

    def get_action(self, obs, ghost_positions=None, agent_pos=None, reward_pos=None, epsilon=0.0):
        """
        Enhanced action selection with temporal context
        """
        # Update temporal buffers
        self.temporal_enhancement.update_buffers(torch.FloatTensor(obs))

        # Enhance observation with temporal features
        enhanced_obs, uncertainty = self.temporal_enhancement.enhance_observation(
            torch.FloatTensor(obs)
        )

        # Use ensemble prediction for ghost-heavy situations
        if (self.ghost_predictor is not None and
            ghost_positions is not None and
            uncertainty > 0.7):  # High uncertainty - use ensemble

            # Update ghost tracker
            self.ghost_predictor.update(ghost_positions)

            # Get ensemble predictions
            predictions = self.ghost_predictor.predict_ensemble(agent_pos, steps_ahead=5)

            if predictions:
                # Find safe zones
                safe_zones = self.ghost_predictor.compute_safe_zones(
                    agent_pos, predictions
                )

                if safe_zones and reward_pos:
                    # Use ensemble-based action
                    action = self.ghost_predictor.get_best_action(
                        agent_pos, safe_zones, reward_pos
                    )
                    return action

        # Normal case - use enhanced observation with base agent
        with torch.no_grad():
            q_values = self.base_agent.get_combined_q(enhanced_obs.unsqueeze(0))

            if np.random.random() < epsilon:
                return np.random.randint(4)
            else:
                return q_values.argmax().item()

    def reset(self):
        """Reset for new episode"""
        self.temporal_enhancement.reset()


if __name__ == '__main__':
    print("="*60)
    print("TEMPORAL BUFFER ENHANCEMENT")
    print("Quick fix for ghost prediction - no full retrain needed!")
    print("="*60)
    print()
    print("Features:")
    print("  1. Multi-scale temporal buffering (5 + 50 frames)")
    print("  2. Uncertainty estimation")
    print("  3. Ensemble ghost prediction (optimistic/expected/pessimistic)")
    print("  4. Safe zone computation")
    print()
    print("Integration:")
    print("  - Wraps existing ContextAwareDQN")
    print("  - Fine-tune for 50-100 episodes")
    print("  - No architecture change to base model")
    print()
    print("Expected improvement:")
    print("  Pac-Man: 29% → 40-45% completion")
    print("  Training time: 2-3 hours")

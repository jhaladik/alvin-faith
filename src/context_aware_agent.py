"""
Context-Aware Temporal Hierarchical Agent

Key Innovation: Adds 3-dimensional context vector to observation
- [1,0,0] = No entities (Snake mode)
- [0,1,0] = Balanced entities (2-3)
- [0,0,1] = High threat (4+)

Input: 95 features = 92 temporal + 3 context
"""
import torch
import torch.nn as nn
import numpy as np
try:
    from core.temporal_agent import TemporalHierarchicalDQN, TemporalTrainer
except ModuleNotFoundError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
    from temporal_agent import TemporalHierarchicalDQN, TemporalTrainer


class ContextAwareDQN(nn.Module):
    """
    Hierarchical DQN with context awareness.

    Same architecture as TemporalHierarchicalDQN but accepts 95-dim input:
    - 92 temporal features
    - 3 context features [no_entities, balanced, high_threat]
    """

    def __init__(self, obs_dim=95, action_dim=4, hidden_dim=128):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Perception network (now sees context)
        self.perception_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Hierarchical Q-heads (same as before)
        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.avoidance_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.positioning_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.collection_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Priority weights (context-dependent learning will adjust these internally)
        self.priority_weights = {
            'survive': 8.0,
            'avoid': 4.0,
            'position': 2.0,
            'collect': 8.0
        }

    def forward(self, x):
        """Forward pass with context"""
        state_understanding = self.perception_net(x)

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
        """Select action using epsilon-greedy"""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.get_combined_q(state_tensor)
            return q_values.argmax(dim=1).item()

    def set_inference_weights(self, survive=8.0, avoid=4.0, position=2.0, collect=8.0):
        """Adjust priority weights at inference time"""
        self.priority_weights = {
            'survive': survive,
            'avoid': avoid,
            'position': position,
            'collect': collect
        }


def infer_context_from_observation(obs):
    """
    Infer context vector from observation features.

    Analyzes entity detection rays (first 24 features: 8 rays * 3 values each)
    to count how many entities are detected.

    Returns:
        [1,0,0] if no entities
        [0,1,0] if 1-3 entities
        [0,0,1] if 4+ entities
    """
    # Count detected entities from rays
    # FIX: Each ray actually has: [reward_dist, entity_dist, wall_dist]
    # (See temporal_observer.py:148-158 - reward first, then entity, then wall)
    # Entity detected if entity_dist < 0.9

    entity_count = 0
    for i in range(8):  # 8 rays
        entity_dist = obs[i * 3 + 1]  # FIX: Entity is at offset +1, not 0!
        if entity_dist < 0.9:  # Entity detected in this direction
            entity_count += 1

    # Map to context vector
    if entity_count == 0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)  # No entities
    elif entity_count <= 3:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Balanced
    else:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)  # High threat


def add_context_to_observation(obs, context):
    """Concatenate context vector to observation"""
    return np.concatenate([obs, context])


def demo():
    """Demonstrate context-aware agent"""
    print("=" * 60)
    print("CONTEXT-AWARE AGENT DEMO")
    print("=" * 60)
    print()

    # Create agent
    agent = ContextAwareDQN(obs_dim=95)

    print(f"Input dimension: {agent.obs_dim} (92 temporal + 3 context)")
    print(f"Total parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print()

    # Test with different contexts
    contexts = {
        'Snake (no entities)': [1.0, 0.0, 0.0],
        'Balanced (2-3 entities)': [0.0, 1.0, 0.0],
        'Survival (4+ entities)': [0.0, 0.0, 1.0]
    }

    # Sample observation
    sample_obs = np.random.randn(92).astype(np.float32)

    for name, context in contexts.items():
        obs_with_context = add_context_to_observation(sample_obs, context)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_with_context).unsqueeze(0)
            q_values = agent.get_combined_q(obs_tensor)
            action = q_values.argmax().item()

        print(f"{name}:")
        print(f"  Context: {context}")
        print(f"  Q-values: {q_values.numpy()[0]}")
        print(f"  Action: {['UP', 'DOWN', 'LEFT', 'RIGHT'][action]}")
        print()

    print("KEY INSIGHT:")
    print("Agent learns DIFFERENT behaviors for DIFFERENT contexts!")
    print("- No entities → aggressive collection")
    print("- Balanced → tactical positioning")
    print("- High threat → survival priority")


if __name__ == '__main__':
    demo()

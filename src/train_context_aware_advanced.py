"""
Advanced Context-Aware Training with Enhancements

NEW FEATURES:
1. World Model Planning - Uses trained world model for lookahead
2. Prioritized Experience Replay - Samples important transitions more frequently
3. Q-Head Dominance Analysis - Tracks which heads dominate in each context

Usage:
    python train_context_aware_advanced.py --episodes 2000
    python train_context_aware_advanced.py --episodes 2000 --use-planning --planning-horizon 3
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

import torch
import numpy as np
import argparse
from datetime import datetime
from collections import deque
import random
from context_aware_agent import (
    ContextAwareDQN,
    infer_context_from_observation,
    add_context_to_observation
)
from core.temporal_env import TemporalRandom2DEnv
from core.world_model import WorldModelNetwork


class ContinuousMotivationRewardSystem:
    """
    Comprehensive reward system with 5 components for continuous motivation:

    1. APPROACH GRADIENT: Reward getting closer to pellets (+0.5 per tile)
    2. COMBO SYSTEM: Increasing rewards for chaining collections (2x, 4x, 6x...)
    3. RISK MULTIPLIER: 3x reward for collecting near enemies (dist < 3)
    4. SURVIVAL STREAK: Milestone bonuses at 50, 100, 200, 300 steps alive
    5. LEVEL PROGRESSION: Complete levels for exponential bonuses (100, 200, 400, 800)

    Each component targets different Q-heads:
    - Approach + Combo → COLLECT head
    - Risk multiplier → AVOID + COLLECT synergy
    - Survival streak → SURVIVE + AVOID heads
    - Levels → Long-term POSITION + PLANNING
    """

    def __init__(self, context_name='balanced'):
        self.context_name = context_name
        self.reset()

        # Level configuration per context
        self.level_configs = {
            'snake': {
                1: {'pellets': 10, 'enemies': 0, 'completion_bonus': 100},
                2: {'pellets': 15, 'enemies': 0, 'completion_bonus': 200},
                3: {'pellets': 20, 'enemies': 1, 'completion_bonus': 400},
                4: {'pellets': 25, 'enemies': 1, 'completion_bonus': 800},
                5: {'pellets': 30, 'enemies': 2, 'completion_bonus': 1600},
            },
            'balanced': {
                1: {'pellets': 10, 'enemies': 2, 'completion_bonus': 100},
                2: {'pellets': 15, 'enemies': 3, 'completion_bonus': 200},
                3: {'pellets': 20, 'enemies': 4, 'completion_bonus': 400},
                4: {'pellets': 25, 'enemies': 5, 'completion_bonus': 800},
                5: {'pellets': 30, 'enemies': 6, 'completion_bonus': 1600},
            },
            'survival': {
                1: {'pellets': 8, 'enemies': 4, 'completion_bonus': 150},
                2: {'pellets': 12, 'enemies': 5, 'completion_bonus': 300},
                3: {'pellets': 15, 'enemies': 6, 'completion_bonus': 600},
                4: {'pellets': 20, 'enemies': 7, 'completion_bonus': 1200},
                5: {'pellets': 25, 'enemies': 8, 'completion_bonus': 2400},
            }
        }

    def reset(self):
        """Reset for new episode"""
        self.combo_count = 0
        self.combo_timer = 0
        self.steps_alive = 0
        self.last_pellet_dist = None
        self.pellets_collected_this_episode = 0
        self.episode_start_step = 0

    def get_current_level_config(self, level):
        """Get configuration for current level"""
        config = self.level_configs[self.context_name]
        return config.get(level, config[5])  # Default to level 5 if higher

    def calculate_reward(self, env_reward, info, nearest_pellet_dist, nearest_enemy_dist):
        """
        Calculate enhanced reward based on all 5 components

        Args:
            env_reward: Original reward from environment
            info: Info dict from environment step
            nearest_pellet_dist: Distance to nearest pellet
            nearest_enemy_dist: Distance to nearest enemy

        Returns:
            total_reward: Enhanced reward
            reward_breakdown: Dict with component rewards for logging
        """
        total_reward = 0.0
        breakdown = {
            'env': env_reward,
            'approach': 0.0,
            'combo': 0.0,
            'risk': 0.0,
            'streak': 0.0,
            'level': 0.0,
            'death_penalty': 0.0
        }

        # COMPONENT 1: APPROACH GRADIENT
        # Continuous feedback for moving toward pellets
        if nearest_pellet_dist is not None and self.last_pellet_dist is not None:
            if nearest_pellet_dist < self.last_pellet_dist:
                breakdown['approach'] = 0.5  # Getting closer
            else:
                breakdown['approach'] = -0.1  # Getting farther
        self.last_pellet_dist = nearest_pellet_dist

        # COMPONENT 2: COMBO SYSTEM + COMPONENT 3: RISK MULTIPLIER
        # Combo builds with each collection, risk multiplies reward
        collected = info.get('collected_reward', False)
        if collected:
            self.pellets_collected_this_episode += 1
            self.combo_count += 1

            # Base combo reward (increases with chain)
            base_pellet_reward = 10.0
            combo_bonus = self.combo_count * 2.0  # +2, +4, +6, +8...
            pellet_reward = base_pellet_reward + combo_bonus

            # Risk multiplier (3x if collecting near enemies)
            risk_mult = 1.0
            if nearest_enemy_dist is not None:
                if nearest_enemy_dist < 3.0:
                    risk_mult = 3.0  # Very risky!
                    breakdown['risk'] = pellet_reward * 2.0  # Extra from multiplier
                elif nearest_enemy_dist < 5.0:
                    risk_mult = 2.0
                    breakdown['risk'] = pellet_reward * 1.0

            breakdown['combo'] = pellet_reward * risk_mult

            # Reset combo timer (have 20 steps to get next pellet)
            self.combo_timer = 20
        else:
            # Combo timer decay
            if self.combo_timer > 0:
                self.combo_timer -= 1
                if self.combo_timer <= 0:
                    # Combo broken!
                    breakdown['combo'] = -5.0
                    self.combo_count = 0

        # COMPONENT 4: SURVIVAL STREAK
        # Milestone bonuses and continuous survival reward
        self.steps_alive += 1

        # Continuous survival reward
        breakdown['streak'] = 0.05

        # Milestone bonuses (exponential)
        milestones = [50, 100, 200, 300, 400, 500]
        for milestone in milestones:
            if self.steps_alive == milestone:
                milestone_bonus = milestone * 0.5  # +25, +50, +100, +150...
                breakdown['streak'] += milestone_bonus
                break

        # Death penalty (lose streak progress)
        if info.get('died', False):
            streak_loss = self.steps_alive * 0.1
            breakdown['death_penalty'] = -(50.0 + streak_loss)
            breakdown['streak'] = 0  # No streak reward this step
            self.steps_alive = 0
            self.combo_count = 0
            self.combo_timer = 0

        # Sum all components
        total_reward = sum(breakdown.values())

        return total_reward, breakdown

    def get_stats(self):
        """Get current stats for logging"""
        return {
            'combo': self.combo_count,
            'combo_timer': self.combo_timer,
            'streak': self.steps_alive,
            'pellets': self.pellets_collected_this_episode
        }


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer

    Samples transitions based on TD-error priority.
    High priority = agent was surprised = important to learn from
    """

    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Args:
            capacity: Max buffer size
            alpha: Priority exponent (0=uniform, 1=fully prioritized)
            beta: Importance sampling correction (0=no correction, 1=full)
            beta_increment: Increase beta to 1.0 over training
        """
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use
        self.beta = beta    # Importance-sampling correction
        self.beta_increment = beta_increment

        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def add(self, transition, td_error=None):
        """Add transition with priority"""
        # New transitions get max priority (will be sampled soon)
        if td_error is None:
            max_priority = max(self.priorities) if self.priorities else 1.0
        else:
            max_priority = abs(td_error) + 1e-5  # Small epsilon prevents zero priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample batch with priorities"""
        if len(self.buffer) < batch_size:
            return None, None, None

        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # Importance sampling weights (for correcting bias)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Get transitions
        transitions = [self.buffer[i] for i in indices]

        # Increase beta over time (more correction as training progresses)
        self.beta = min(1.0, self.beta + self.beta_increment)

        return transitions, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities for sampled transitions"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-5
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class WorldModelPlanner:
    """
    Uses world model to plan ahead before taking actions

    Simulates multiple action sequences and picks the one with best expected return.
    """

    def __init__(self, policy_net, world_model, gamma=0.99, num_rollouts=5, horizon=3):
        """
        Args:
            policy_net: Policy network to evaluate actions
            world_model: World model to simulate futures
            gamma: Discount factor
            num_rollouts: Number of trajectories to simulate per action
            horizon: How many steps to look ahead
        """
        self.policy_net = policy_net
        self.world_model = world_model
        self.gamma = gamma
        self.num_rollouts = num_rollouts
        self.horizon = horizon

    def plan_action(self, state):
        """
        Plan best action by simulating futures with world model

        Returns:
            best_action: Action with highest expected return
            expected_return: Estimated value of best action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        best_action = None
        best_return = -float('inf')

        # Try each action
        for action in range(4):  # 4 actions: UP, DOWN, LEFT, RIGHT
            total_return = 0.0

            # Monte Carlo: simulate multiple rollouts
            for _ in range(self.num_rollouts):
                rollout_return = self._simulate_rollout(state_tensor, action)
                total_return += rollout_return

            avg_return = total_return / self.num_rollouts

            if avg_return > best_return:
                best_return = avg_return
                best_action = action

        return best_action, best_return

    def _simulate_rollout(self, state, first_action):
        """Simulate one trajectory using world model"""
        current_state = state.clone()
        total_return = 0.0
        discount = 1.0

        with torch.no_grad():
            # Take first action
            action_tensor = torch.LongTensor([first_action])
            next_state, reward, done = self.world_model(current_state, action_tensor)
            total_return += reward.item() * discount
            discount *= self.gamma

            if done.item() > 0.5:
                return total_return

            current_state = next_state

            # Simulate remaining horizon steps using policy
            for _ in range(self.horizon - 1):
                # Use policy to select action
                q_values = self.policy_net.get_combined_q(current_state)
                action = q_values.argmax(dim=1).item()

                # Simulate with world model
                action_tensor = torch.LongTensor([action])
                next_state, reward, done = self.world_model(current_state, action_tensor)

                total_return += reward.item() * discount
                discount *= self.gamma

                if done.item() > 0.5:
                    break

                current_state = next_state

        return total_return


class QHeadAnalyzer:
    """
    Analyzes which Q-heads are dominant in different contexts

    Tracks:
    - Which head has highest Q-values
    - Which head changes decision most
    - Per-context head dominance
    """

    def __init__(self):
        self.dominance_counts = {
            'snake': {'survive': 0, 'avoid': 0, 'position': 0, 'collect': 0},
            'balanced': {'survive': 0, 'avoid': 0, 'position': 0, 'collect': 0},
            'survival': {'survive': 0, 'avoid': 0, 'position': 0, 'collect': 0}
        }

        self.q_value_sums = {
            'snake': {'survive': [], 'avoid': [], 'position': [], 'collect': []},
            'balanced': {'survive': [], 'avoid': [], 'position': [], 'collect': []},
            'survival': {'survive': [], 'avoid': [], 'position': [], 'collect': []}
        }

    def analyze_step(self, policy_net, state, context_name):
        """Analyze Q-head dominance for current state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_dict = policy_net.forward(state_tensor)

            # Get Q-values for selected action
            combined_q = policy_net.get_combined_q(state_tensor)
            best_action = combined_q.argmax(dim=1).item()

            # Record Q-values
            for head_name in ['survive', 'avoid', 'position', 'collect']:
                q_val = q_dict[head_name][0, best_action].item()
                self.q_value_sums[context_name][head_name].append(q_val)

            # Determine dominant head (highest Q-value)
            head_q_values = {
                head: q_dict[head][0, best_action].item()
                for head in ['survive', 'avoid', 'position', 'collect']
            }
            dominant_head = max(head_q_values, key=head_q_values.get)
            self.dominance_counts[context_name][dominant_head] += 1

    def get_summary(self):
        """Get dominance analysis summary"""
        summary = {}

        for context in ['snake', 'balanced', 'survival']:
            total = sum(self.dominance_counts[context].values())
            if total == 0:
                continue

            percentages = {
                head: (count / total * 100)
                for head, count in self.dominance_counts[context].items()
            }

            avg_q_values = {
                head: np.mean(values) if values else 0.0
                for head, values in self.q_value_sums[context].items()
            }

            summary[context] = {
                'dominance_pct': percentages,
                'avg_q_values': avg_q_values
            }

        return summary


class AdvancedContextAwareTrainer:
    """
    Enhanced trainer with:
    1. Prioritized replay
    2. World model planning
    3. Q-head analysis
    """

    def __init__(
        self,
        env_size=20,
        num_rewards=10,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        lr_policy=0.0001,
        lr_world_model=0.0003,
        target_update_freq=500,
        use_planning=False,
        planning_freq=0.2,
        planning_horizon=3
    ):
        self.env_size = env_size
        self.num_rewards = num_rewards
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        # Planning parameters
        self.use_planning = use_planning
        self.planning_freq = planning_freq

        # Context distribution
        self.context_distribution = {
            'snake': 0.30,
            'balanced': 0.50,
            'survival': 0.20
        }

        self.context_configs = {
            'snake': (0, 0),
            'balanced': (2, 3),
            'survival': (4, 6)
        }

        # Networks
        self.policy_net = ContextAwareDQN(obs_dim=95, action_dim=4)
        self.target_net = ContextAwareDQN(obs_dim=95, action_dim=4)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.world_model = WorldModelNetwork(state_dim=95, action_dim=4)

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=lr_world_model)

        # ENHANCEMENT 1: Prioritized Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)

        # ENHANCEMENT 2: World Model Planner
        if self.use_planning:
            self.planner = WorldModelPlanner(
                self.policy_net,
                self.world_model,
                gamma=gamma,
                num_rollouts=5,
                horizon=planning_horizon
            )

        # ENHANCEMENT 3: Q-Head Analyzer
        self.q_head_analyzer = QHeadAnalyzer()

        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.world_model_losses = []
        self.steps_done = 0
        self.planning_count = 0
        self.reactive_count = 0

        # Context tracking
        self.context_episode_counts = {'snake': 0, 'balanced': 0, 'survival': 0}
        self.context_avg_rewards = {'snake': [], 'balanced': [], 'survival': []}

        # ENHANCEMENT 4: Level Progression & Continuous Motivation
        self.context_levels = {'snake': 1, 'balanced': 1, 'survival': 1}
        self.reward_systems = {
            'snake': ContinuousMotivationRewardSystem('snake'),
            'balanced': ContinuousMotivationRewardSystem('balanced'),
            'survival': ContinuousMotivationRewardSystem('survival')
        }
        self.level_completions = {'snake': 0, 'balanced': 0, 'survival': 0}

        print("=" * 60)
        print("ADVANCED CONTEXT-AWARE TRAINER INITIALIZED")
        print("=" * 60)
        print(f"Policy network: {sum(p.numel() for p in self.policy_net.parameters()):,} parameters")
        print(f"World model: {sum(p.numel() for p in self.world_model.parameters()):,} parameters")
        print()
        print("ENHANCEMENTS:")
        print(f"  [1] Prioritized Replay: YES (alpha=0.6, beta=0.4->1.0)")
        print(f"  [2] World Model Planning: {'YES' if use_planning else 'NO'}")
        if use_planning:
            print(f"      - Planning frequency: {planning_freq*100:.0f}%")
            print(f"      - Planning horizon: {planning_horizon} steps")
        print(f"  [3] Q-Head Analysis: YES (tracks dominance per context)")
        print(f"  [4] Continuous Motivation: YES (5 reward components)")
        print(f"      - Approach gradient, Combo system, Risk multiplier")
        print(f"      - Survival streaks, Level progression")
        print()
        print("CONTEXT DISTRIBUTION & STARTING LEVELS:")
        for context, prob in self.context_distribution.items():
            level = self.context_levels[context]
            config = self.reward_systems[context].get_current_level_config(level)
            print(f"  {context:8s}: {prob*100:4.0f}% | Level {level}: {config['pellets']} pellets, {config['enemies']} enemies")
        print()

    def sample_context(self):
        """Sample a context based on distribution"""
        contexts = list(self.context_distribution.keys())
        probs = list(self.context_distribution.values())
        return np.random.choice(contexts, p=probs)

    def create_env_for_context(self, context):
        """Create environment based on current level configuration"""
        # Get level configuration
        current_level = self.context_levels[context]
        level_config = self.reward_systems[context].get_current_level_config(current_level)

        # Create environment with level-specific settings
        env = TemporalRandom2DEnv(
            grid_size=(self.env_size, self.env_size),
            num_entities=level_config['enemies'],
            num_rewards=level_config['pellets']
        )
        return env, level_config

    def get_context_vector(self, context):
        """Get one-hot context vector for training"""
        if context == 'snake':
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif context == 'balanced':
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def extract_distances_from_obs(self, obs):
        """
        Extract nearest pellet and enemy distances from observation

        Observation structure (from temporal_observer.py):
        - Ray observations: 8 rays × 3 values = 24 features
          Each ray: [reward_dist, entity_dist, wall_dist]
        - Features 46-47: reward_direction_x, reward_direction_y
        - Feature 40: nearest_entity_distance

        Returns:
            nearest_pellet_dist, nearest_enemy_dist (both 0-1 normalized)
        """
        # Get nearest entity distance (feature 40)
        if len(obs) > 40:
            nearest_enemy_dist = obs[40]
        else:
            nearest_enemy_dist = 1.0  # Max distance (normalized)

        # Get nearest pellet distance from rays (features 0-23)
        # Every 3rd element starting from 0 is reward_dist
        pellet_distances = [obs[i] for i in range(0, min(24, len(obs)), 3)]
        nearest_pellet_dist = min(pellet_distances) if pellet_distances else 1.0

        # Convert from normalized (0=close, 1=far) to actual distances (approx)
        # Multiply by ray_length=10 to get rough tile distance
        nearest_pellet_dist = nearest_pellet_dist * 10.0
        nearest_enemy_dist = nearest_enemy_dist * 10.0

        return nearest_pellet_dist, nearest_enemy_dist

    def train_episode(self, context, epsilon):
        """Train one episode with given context and continuous motivation"""
        env, level_config = self.create_env_for_context(context)

        # ENHANCEMENT 4: Reset reward system for new episode
        reward_system = self.reward_systems[context]
        reward_system.reset()

        obs = env.reset()
        context_vector = self.get_context_vector(context)
        obs_with_context = add_context_to_observation(obs, context_vector)

        episode_reward = 0
        episode_enhanced_reward = 0
        episode_length = 0
        level_completed = False

        done = False
        while not done and episode_length < 1000:
            # ENHANCEMENT 2: World Model Planning (occasionally)
            if self.use_planning and random.random() < self.planning_freq and len(self.replay_buffer) > 1000:
                action, _ = self.planner.plan_action(obs_with_context)
                self.planning_count += 1
            else:
                # Normal epsilon-greedy action
                action = self.policy_net.get_action(obs_with_context, epsilon=epsilon)
                self.reactive_count += 1

            # ENHANCEMENT 3: Q-Head Analysis
            if episode_length % 10 == 0:  # Sample every 10 steps
                self.q_head_analyzer.analyze_step(self.policy_net, obs_with_context, context)

            # Execute action
            next_obs, env_reward, done, info = env.step(action)
            next_obs_with_context = add_context_to_observation(next_obs, context_vector)

            # ENHANCEMENT 4: Calculate enhanced reward with continuous motivation
            # Extract distances for reward calculation
            pellet_dist, enemy_dist = self.extract_distances_from_obs(next_obs)

            # Calculate enhanced reward
            enhanced_reward, reward_breakdown = reward_system.calculate_reward(
                env_reward, info, pellet_dist, enemy_dist
            )

            # Check for level completion (all pellets collected)
            if info.get('rewards_left', 0) == 0 and not level_completed:
                level_completed = True
                current_level = self.context_levels[context]
                completion_bonus = level_config['completion_bonus']
                enhanced_reward += completion_bonus
                reward_breakdown['level'] = completion_bonus

                # Advance to next level
                self.context_levels[context] = min(5, current_level + 1)
                self.level_completions[context] += 1

                # Level completed but don't end episode - spawn new level!
                # Reset environment with new level config
                new_level_config = reward_system.get_current_level_config(self.context_levels[context])
                env = TemporalRandom2DEnv(
                    grid_size=(self.env_size, self.env_size),
                    num_entities=new_level_config['enemies'],
                    num_rewards=new_level_config['pellets']
                )
                next_obs = env.reset()
                next_obs_with_context = add_context_to_observation(next_obs, context_vector)
                done = False  # Continue playing!
                level_completed = False  # Reset for next level

            # Store transition in prioritized replay buffer (use enhanced reward!)
            transition = {
                'state': obs_with_context.copy(),
                'action': action,
                'reward': enhanced_reward,  # IMPORTANT: Use enhanced reward for training!
                'next_state': next_obs_with_context.copy(),
                'done': done
            }
            self.replay_buffer.add(transition)

            # Train networks
            if len(self.replay_buffer) >= self.batch_size:
                policy_loss = self._train_policy_step()
                if policy_loss is not None:
                    self.policy_losses.append(policy_loss)

                world_model_loss = self._train_world_model_step()
                if world_model_loss is not None:
                    self.world_model_losses.append(world_model_loss)

            # Update target network
            if self.steps_done % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            obs_with_context = next_obs_with_context
            episode_reward += env_reward
            episode_enhanced_reward += enhanced_reward
            episode_length += 1
            self.steps_done += 1

        return episode_enhanced_reward, episode_length

    def _train_policy_step(self):
        """Train policy with prioritized replay"""
        # ENHANCEMENT 1: Sample with priorities
        result = self.replay_buffer.sample(self.batch_size)
        if result[0] is None:
            return None

        transitions, indices, is_weights = result

        # Convert to tensors
        states = torch.FloatTensor(np.stack([t['state'] for t in transitions]))
        actions = torch.LongTensor(np.array([t['action'] for t in transitions]))
        rewards = torch.FloatTensor(np.array([t['reward'] for t in transitions]))
        next_states = torch.FloatTensor(np.stack([t['next_state'] for t in transitions]))
        dones = torch.FloatTensor(np.array([t['done'] for t in transitions]))
        is_weights = torch.FloatTensor(is_weights)

        # Current Q-values
        current_q = self.policy_net.get_combined_q(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net.get_combined_q(next_states)
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # TD errors (for priority update)
        td_errors = (target_q - current_q).detach().cpu().numpy()

        # Weighted loss (importance sampling correction)
        loss = (is_weights * torch.nn.functional.mse_loss(current_q, target_q, reduction='none')).mean()

        # Optimize
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()

        # ENHANCEMENT 1: Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)

        return loss.item()

    def _train_world_model_step(self):
        """Train world model (uniform sampling is fine here)"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample uniformly (world model doesn't need prioritization)
        indices = np.random.choice(len(self.replay_buffer.buffer), self.batch_size, replace=False)
        transitions = [self.replay_buffer.buffer[i] for i in indices]

        states = torch.FloatTensor(np.stack([t['state'] for t in transitions]))
        actions = torch.LongTensor(np.array([t['action'] for t in transitions]))
        rewards = torch.FloatTensor(np.array([t['reward'] for t in transitions]))
        next_states = torch.FloatTensor(np.stack([t['next_state'] for t in transitions]))
        dones = torch.FloatTensor(np.array([t['done'] for t in transitions]))

        # Forward pass
        pred_next_states, pred_rewards, pred_dones = self.world_model(states, actions)

        # Compute losses
        state_loss = torch.nn.functional.mse_loss(pred_next_states, next_states)
        reward_loss = torch.nn.functional.mse_loss(pred_rewards.squeeze(), rewards)
        done_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_dones.squeeze(), dones)

        total_loss = state_loss + reward_loss + done_loss

        # Optimize
        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()

        return total_loss.item()

    def train(self, num_episodes, log_every=100):
        """Main training loop"""
        print("=" * 60)
        print("STARTING ADVANCED CONTEXT-AWARE TRAINING")
        print("=" * 60)
        print()

        # Track starting episode (for resumption)
        start_episode = len(self.episode_rewards)
        total_episodes = start_episode + num_episodes

        if start_episode > 0:
            print(f"Resuming from episode {start_episode}")
            print(f"Will train {num_episodes} additional episodes (to episode {total_episodes})")
            print()

        best_avg_reward = -float('inf')

        for episode in range(start_episode, total_episodes):
            context = self.sample_context()
            self.context_episode_counts[context] += 1

            # Epsilon decay (based on total episodes, not just new ones)
            epsilon = max(0.01, 1.0 - episode / (total_episodes * 0.5))

            # Train episode
            reward, length = self.train_episode(context, epsilon)

            # Track stats
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            self.context_avg_rewards[context].append(reward)

            # Logging
            if (episode + 1) % log_every == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                avg_policy_loss = np.mean(self.policy_losses[-100:]) if self.policy_losses else 0
                avg_wm_loss = np.mean(self.world_model_losses[-100:]) if self.world_model_losses else 0

                print(f"Episode {episode+1}/{num_episodes}")
                print(f"  Avg Reward (100): {avg_reward:.2f}")
                print(f"  Avg Length (100): {avg_length:.1f}")
                print(f"  Policy Loss: {avg_policy_loss:.4f}")
                print(f"  World Model Loss: {avg_wm_loss:.4f}")
                print(f"  Epsilon: {epsilon:.3f}")
                print(f"  Buffer Size: {len(self.replay_buffer)}")
                print(f"  Steps: {self.steps_done}")

                if self.use_planning:
                    total_actions = self.planning_count + self.reactive_count
                    plan_pct = (self.planning_count / total_actions * 100) if total_actions > 0 else 0
                    print(f"  Planning: {plan_pct:.1f}% ({self.planning_count}/{total_actions})")

                # Context breakdown with level progression
                print("  Context Distribution & Levels:")
                for ctx in ['snake', 'balanced', 'survival']:
                    count = self.context_episode_counts[ctx]
                    pct = (count / (episode + 1)) * 100
                    ctx_avg = np.mean(self.context_avg_rewards[ctx][-50:]) if self.context_avg_rewards[ctx] else 0
                    level = self.context_levels[ctx]
                    completions = self.level_completions[ctx]

                    # Get current level config
                    level_config = self.reward_systems[ctx].get_current_level_config(level)
                    print(f"    {ctx:8s}: {count:4d} episodes ({pct:4.1f}%) - avg reward: {ctx_avg:6.2f}")
                    print(f"              Level {level} ({level_config['pellets']}p/{level_config['enemies']}e) - {completions} completions")

                # ENHANCEMENT 4: Continuous Motivation Stats
                print("\n  Reward System Stats (current episode):")
                for ctx in ['snake', 'balanced', 'survival']:
                    stats = self.reward_systems[ctx].get_stats()
                    print(f"    {ctx:8s}: Combo={stats['combo']}, Streak={stats['streak']}, Pellets={stats['pellets']}")

                # ENHANCEMENT 3: Q-Head Dominance Analysis
                if (episode + 1) % (log_every * 2) == 0:
                    print("\n  Q-HEAD DOMINANCE ANALYSIS:")
                    summary = self.q_head_analyzer.get_summary()
                    for context, data in summary.items():
                        print(f"    {context.upper()}:")
                        for head, pct in data['dominance_pct'].items():
                            avg_q = data['avg_q_values'][head]
                            print(f"      {head:8s}: {pct:5.1f}% dominant | avg Q={avg_q:6.2f}")

                print()

                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.save(f"checkpoints/context_aware_advanced_{timestamp}_best")
                    print(f"  [BEST] Saved model (avg reward: {avg_reward:.2f})")
                    print()

        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total episodes trained: {len(self.episode_rewards)}")
        print(f"Episodes this session: {num_episodes}")
        print(f"Best avg reward: {best_avg_reward:.2f}")
        print(f"Final epsilon: {epsilon:.3f}")
        print()

        # Final Q-head analysis
        print("FINAL Q-HEAD DOMINANCE:")
        summary = self.q_head_analyzer.get_summary()
        for context, data in summary.items():
            print(f"  {context.upper()}:")
            sorted_heads = sorted(data['dominance_pct'].items(), key=lambda x: x[1], reverse=True)
            for head, pct in sorted_heads:
                avg_q = data['avg_q_values'][head]
                print(f"    {head:8s}: {pct:5.1f}% | avg Q={avg_q:6.2f}")

        # Final context summary with level progression
        print("\nFINAL CONTEXT PERFORMANCE & LEVELS:")
        for ctx in ['snake', 'balanced', 'survival']:
            count = self.context_episode_counts[ctx]
            avg = np.mean(self.context_avg_rewards[ctx][-100:]) if len(self.context_avg_rewards[ctx]) >= 100 else np.mean(self.context_avg_rewards[ctx])
            level = self.context_levels[ctx]
            completions = self.level_completions[ctx]
            print(f"  {ctx:8s}: {count:4d} episodes - avg reward: {avg:6.2f}")
            print(f"            Final Level: {level} - Total Completions: {completions}")

    def save(self, base_path):
        """Save models with Q-head analysis"""
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        # Save policy
        policy_checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.policy_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'context_episode_counts': self.context_episode_counts,
            'context_avg_rewards': self.context_avg_rewards,
            'steps_done': self.steps_done,
            'q_head_analysis': self.q_head_analyzer.get_summary(),
            'planning_count': self.planning_count,
            'reactive_count': self.reactive_count,
            'context_levels': self.context_levels,  # NEW
            'level_completions': self.level_completions  # NEW
        }
        torch.save(policy_checkpoint, f"{base_path}_policy.pth")

        # Save world model
        world_model_checkpoint = {
            'model': self.world_model.state_dict(),
            'optimizer': self.world_model_optimizer.state_dict(),
            'losses': self.world_model_losses
        }
        torch.save(world_model_checkpoint, f"{base_path}_world_model.pth")

        print(f"Saved checkpoint: {base_path}")

    def load(self, policy_path, world_model_path=None):
        """Load checkpoint to resume training"""
        print(f"Loading checkpoint: {policy_path}")

        # Load policy checkpoint
        checkpoint = torch.load(policy_path, map_location='cpu', weights_only=False)

        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.policy_optimizer.load_state_dict(checkpoint['optimizer'])

        # Restore training state
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.context_episode_counts = checkpoint.get('context_episode_counts',
                                                      {'snake': 0, 'balanced': 0, 'survival': 0})
        self.context_avg_rewards = checkpoint.get('context_avg_rewards',
                                                   {'snake': [], 'balanced': [], 'survival': []})
        self.steps_done = checkpoint.get('steps_done', 0)
        self.planning_count = checkpoint.get('planning_count', 0)
        self.reactive_count = checkpoint.get('reactive_count', 0)
        self.context_levels = checkpoint.get('context_levels', {'snake': 1, 'balanced': 1, 'survival': 1})
        self.level_completions = checkpoint.get('level_completions', {'snake': 0, 'balanced': 0, 'survival': 0})

        episodes_trained = len(self.episode_rewards)
        print(f"  Resuming from episode {episodes_trained}")
        print(f"  Steps completed: {self.steps_done}")
        print(f"  Avg reward (last 100): {np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards):.2f}")
        print(f"  Levels: Snake={self.context_levels['snake']}, Balanced={self.context_levels['balanced']}, Survival={self.context_levels['survival']}")
        print(f"  Completions: Snake={self.level_completions['snake']}, Balanced={self.level_completions['balanced']}, Survival={self.level_completions['survival']}")

        # Load world model if provided
        if world_model_path and os.path.exists(world_model_path):
            print(f"Loading world model: {world_model_path}")
            wm_checkpoint = torch.load(world_model_path, map_location='cpu', weights_only=False)
            self.world_model.load_state_dict(wm_checkpoint['model'])
            self.world_model_optimizer.load_state_dict(wm_checkpoint['optimizer'])
            self.world_model_losses = wm_checkpoint.get('losses', [])
            print(f"  World model losses: {len(self.world_model_losses)} recorded")
        else:
            # Try to find world model with same base name
            base_path = policy_path.replace('_policy.pth', '')
            auto_wm_path = f"{base_path}_world_model.pth"
            if os.path.exists(auto_wm_path):
                print(f"Auto-loading world model: {auto_wm_path}")
                wm_checkpoint = torch.load(auto_wm_path, map_location='cpu', weights_only=False)
                self.world_model.load_state_dict(wm_checkpoint['model'])
                self.world_model_optimizer.load_state_dict(wm_checkpoint['optimizer'])
                self.world_model_losses = wm_checkpoint.get('losses', [])
                print(f"  World model loaded successfully")

        print()


def main():
    parser = argparse.ArgumentParser(description='Train Advanced Context-Aware Agent')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of episodes')
    parser.add_argument('--log-every', type=int, default=100, help='Log frequency')
    parser.add_argument('--env-size', type=int, default=20, help='Environment size')
    parser.add_argument('--num-rewards', type=int, default=10, help='Number of rewards')
    parser.add_argument('--use-planning', action='store_true', help='Enable world model planning')
    parser.add_argument('--planning-freq', type=float, default=0.2, help='Planning frequency (0-1)')
    parser.add_argument('--planning-horizon', type=int, default=3, help='Planning lookahead steps')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (path to _policy.pth file)')

    args = parser.parse_args()

    print("=" * 60)
    print("ADVANCED CONTEXT-AWARE AGENT TRAINING")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Log every: {args.log_every}")
    print(f"Environment: {args.env_size}x{args.env_size}")
    print(f"Rewards: {args.num_rewards}")
    print(f"World Model Planning: {'ENABLED' if args.use_planning else 'DISABLED'}")
    if args.use_planning:
        print(f"  Planning frequency: {args.planning_freq*100:.0f}%")
        print(f"  Planning horizon: {args.planning_horizon} steps")
    print()

    # Create trainer
    trainer = AdvancedContextAwareTrainer(
        env_size=args.env_size,
        num_rewards=args.num_rewards,
        use_planning=args.use_planning,
        planning_freq=args.planning_freq,
        planning_horizon=args.planning_horizon
    )

    # Load checkpoint if resuming
    if args.resume:
        trainer.load(args.resume)
        print("RESUMING TRAINING WITH NEW CONFIGURATION")
        if args.use_planning:
            print(f"  Planning NOW ENABLED: {args.planning_freq*100:.0f}% frequency, horizon {args.planning_horizon}")
        print()

    # Train
    trainer.train(num_episodes=args.episodes, log_every=args.log_every)

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save(f"checkpoints/context_aware_advanced_{timestamp}_final")


if __name__ == '__main__':
    main()

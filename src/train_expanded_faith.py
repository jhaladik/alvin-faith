"""
EXPANDED Faith-Based Evolutionary Training System - Option A Implementation

REVOLUTIONARY ARCHITECTURE + EXPANDED SPATIAL-TEMPORAL CAPACITY:
Combines 5 cutting-edge innovations to break through RL plateaus:

1. FAITH PATTERN EVOLUTION
   - Population of 20 behavioral patterns that persist despite negative feedback
   - Genetic algorithm evolution: selection, crossover, mutation
   - Discovers hidden mechanics that Q-learning would never find

2. ENTITY DISCOVERY
   - Learns what entities ARE without being told
   - Discovers: "Pattern #3 = reward giver", "Pattern #5 = chaser"
   - Enables true zero-shot transfer across environments

3. UNIVERSAL PATTERN EXTRACTION
   - Extracts abstract patterns: chase-escape, collection chains, periodic spawns
   - Transfers knowledge: Pac-Man ghosts → Dungeon monsters → Warehouse supervisor
   - Game-agnostic behavioral strategies

4. MECHANIC HYPOTHESIS TESTING
   - Actively discovers hidden game rules
   - Detects: thresholds, timing cycles, action sequences, accumulation effects
   - Superhuman discovery of non-obvious mechanics

5. EXPANDED SPATIAL-TEMPORAL UNDERSTANDING (NEW!)
   - 16 rays × 15 tiles (2x spatial coverage vs baseline)
   - Multi-scale temporal: Micro (5) + Meso (20) + Macro (50) frames
   - Ghost behavior mode detection (chase/scatter/random)
   - 180-dim observations (vs 92 baseline)
   - 20-step planning horizon (vs 5 baseline)

KEY IMPROVEMENTS OVER BASELINE:
- See ghosts approaching 50% earlier (15 vs 10 tiles)
- Detect ghost behavior modes over 20-step patterns
- Plan 4x further ahead (20 vs 5 steps)
- Expected: 35-50 avg Pac-Man score (vs 23.12 baseline)

Usage:
    python train_expanded_faith.py --episodes 500
    python train_expanded_faith.py --episodes 500 --planning-horizon 20 --faith-freq 0.05
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

# Import base system
from train_context_aware_advanced import (
    AdvancedContextAwareTrainer,
    ContinuousMotivationRewardSystem,
    PrioritizedReplayBuffer
)

# Import revolutionary modules
from core.faith_system import FaithPattern, FaithPopulation
from core.entity_discovery import EntityDiscoveryWorldModel, EntityBehaviorLearner
from core.pattern_transfer import UniversalPatternExtractor
from core.mechanic_detectors import MechanicDetector

from context_aware_agent import add_context_to_observation
from core.expanded_temporal_env import ExpandedTemporalRandom2DEnv


class FaithBasedEvolutionaryTrainer(AdvancedContextAwareTrainer):
    """
    Evolutionary trainer with faith-based exploration and entity discovery.

    Extends AdvancedContextAwareTrainer with:
    - Faith pattern evolution (population of 20 patterns)
    - Entity discovery world model (learns what entities are)
    - Universal pattern extraction (transferable strategies)
    - Mechanic hypothesis testing (discovers hidden rules)
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
        lr_entity_discovery=0.0003,
        target_update_freq=500,
        use_planning=True,
        planning_freq=0.2,
        planning_horizon=20,  # EXPANDED: 20 steps vs 5 (4x longer horizon)
        # Faith evolution parameters
        faith_freq=0.3,
        faith_population_size=20,
        evolution_freq=50,  # Evolve population every N episodes
        # Entity discovery parameters
        max_entity_types=20
    ):
        # Initialize base trainer
        super().__init__(
            env_size=env_size,
            num_rewards=num_rewards,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            lr_policy=lr_policy,
            lr_world_model=lr_world_model,
            target_update_freq=target_update_freq,
            use_planning=use_planning,
            planning_freq=planning_freq,
            planning_horizon=planning_horizon
        )

        # EXPANDED: Recreate networks with 183-dim observations (180 + 3 context)
        from context_aware_agent import ContextAwareDQN
        from core.world_model import WorldModelNetwork

        obs_dim_expanded = 183  # 180 (expanded observer) + 3 (context features)

        # Recreate policy networks with expanded dimensions
        self.policy_net = ContextAwareDQN(obs_dim=obs_dim_expanded, action_dim=4)
        self.target_net = ContextAwareDQN(obs_dim=obs_dim_expanded, action_dim=4)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Recreate world model with expanded dimensions
        self.world_model = WorldModelNetwork(state_dim=obs_dim_expanded, action_dim=4)

        # Recreate optimizers with new networks
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=lr_world_model)

        # Recreate planner with expanded world model
        if self.use_planning:
            from train_context_aware_advanced import WorldModelPlanner
            self.planner = WorldModelPlanner(
                self.policy_net,
                self.world_model,
                gamma=gamma,
                num_rollouts=5,
                horizon=planning_horizon
            )

        # REVOLUTION 1: Faith Pattern Evolution
        self.faith_freq = faith_freq
        self.evolution_freq = evolution_freq
        self.faith_population = FaithPopulation(
            population_size=faith_population_size,
            signature_dim=32
        )
        self.faith_action_count = 0
        self.faith_discoveries = []  # Novel rewards found via faith

        # REVOLUTION 2: Entity Discovery World Model
        self.entity_world_model = EntityDiscoveryWorldModel(
            obs_dim=183,  # EXPANDED: 180 (expanded observer) + 3 (context features)
            action_dim=4,
            max_entity_types=max_entity_types
        )
        self.entity_optimizer = torch.optim.Adam(
            self.entity_world_model.parameters(),
            lr=lr_entity_discovery
        )
        self.entity_learner = EntityBehaviorLearner()

        # REVOLUTION 3: Universal Pattern Extraction
        self.pattern_extractors = {
            'snake': UniversalPatternExtractor(),
            'balanced': UniversalPatternExtractor(),
            'survival': UniversalPatternExtractor()
        }
        self.discovered_patterns = {
            'snake': {},
            'balanced': {},
            'survival': {}
        }

        # REVOLUTION 4: Mechanic Detection
        self.mechanic_detectors = {
            'snake': MechanicDetector(),
            'balanced': MechanicDetector(),
            'survival': MechanicDetector()
        }
        self.confirmed_mechanics = {
            'snake': {},
            'balanced': {},
            'survival': {}
        }

        # Tracking
        self.active_faith_pattern = None
        self.current_episode_faith_reward = 0
        self.baseline_reward = 0  # For measuring faith discoveries

        print("=" * 70)
        print("FAITH-BASED EVOLUTIONARY TRAINER INITIALIZED")
        print("=" * 70)
        print()
        print("REVOLUTIONARY COMPONENTS:")
        print(f"  [1] Faith Pattern Evolution:")
        print(f"      - Population: {faith_population_size} patterns")
        print(f"      - Frequency: {faith_freq*100:.0f}% of actions")
        print(f"      - Evolution: Every {evolution_freq} episodes")
        print()
        print(f"  [2] Entity Discovery World Model:")
        print(f"      - Max entity types: {max_entity_types}")
        print(f"      - Parameters: {sum(p.numel() for p in self.entity_world_model.parameters()):,}")
        print(f"      - Learns entities WITHOUT labels")
        print()
        print(f"  [3] Universal Pattern Extraction:")
        print(f"      - 5 pattern types: chase-escape, collection chains, periodic spawns,")
        print(f"        proximity triggers, transformations")
        print(f"      - Enables cross-environment transfer")
        print()
        print(f"  [4] Mechanic Hypothesis Testing:")
        print(f"      - 5 hypothesis types: thresholds, timing cycles, action sequences,")
        print(f"        accumulation, proximity combinations")
        print(f"      - Discovers hidden game rules")
        print()

        # Show expanded observer info ONCE
        print("=" * 70)
        print("EXPANDED SPATIAL-TEMPORAL OBSERVER")
        print("=" * 70)
        print(f"  Rays: 16 (angular resolution: 22°) vs 8 baseline")
        print(f"  Ray length: 15 tiles vs 10 baseline (+50% vision range)")
        print(f"  Total observation: 180 dims vs 92 baseline (+96%)")
        print(f"  Multi-scale temporal: Micro (5) + Meso (20) + Macro (50) frames")
        print(f"  Planning horizon: {planning_horizon} steps vs 5 baseline (4x longer)")
        print()

    def create_env_for_context(self, context):
        """
        Create EXPANDED environment based on current level configuration.

        OVERRIDE: Uses ExpandedTemporalRandom2DEnv instead of TemporalRandom2DEnv
        to provide 180-dim observations (vs 92-dim).
        """
        # Get level configuration
        current_level = self.context_levels[context]
        level_config = self.reward_systems[context].get_current_level_config(current_level)

        # Create EXPANDED environment with level-specific settings
        env = ExpandedTemporalRandom2DEnv(  # EXPANDED: 180-dim observations!
            grid_size=(self.env_size, self.env_size),
            num_entities=level_config['enemies'],
            num_rewards=level_config['pellets']
        )
        return env, level_config

    def train_episode(self, context, epsilon):
        """
        Train one episode with faith-based evolutionary exploration.

        ENHANCEMENT: Integrates all 4 revolutionary systems during training.
        """
        env, level_config = self.create_env_for_context(context)

        # Reset systems for new episode
        reward_system = self.reward_systems[context]
        reward_system.reset()

        # Select faith pattern for this episode
        self.active_faith_pattern = self.faith_population.select_pattern_for_episode()
        self.current_episode_faith_reward = 0
        self.baseline_reward = 0

        obs = env.reset()
        context_vector = self.get_context_vector(context)
        obs_with_context = add_context_to_observation(obs, context_vector)

        episode_reward = 0
        episode_enhanced_reward = 0
        episode_length = 0
        level_completed = False

        # Episode tracking
        faith_actions_this_episode = 0
        planning_actions_this_episode = 0
        reactive_actions_this_episode = 0
        entity_discoveries_this_episode = []

        done = False
        while not done and episode_length < 1000:
            # REVOLUTION 2: Entity Discovery
            # Detect entities in current observation
            obs_tensor = torch.FloatTensor(obs_with_context).unsqueeze(0)
            entity_probs, detected_entities = self.entity_world_model.detect_entities(obs_tensor)

            # Record entities for pattern extraction
            self.pattern_extractors[context].observe(detected_entities, obs)
            self.entity_learner.observe(episode_length, detected_entities)

            # ACTION SELECTION: Faith → Planning → Reactive
            action_source = "reactive"

            # OPTION 1: Faith-based action (30% by default)
            if random.random() < self.faith_freq:
                # Get base Q-action first
                base_action = self.policy_net.get_action(obs_with_context, epsilon=0)

                # Faith pattern modifies action
                if self.active_faith_pattern.should_override(episode_length, obs_tensor):
                    action = self.active_faith_pattern.get_action(
                        obs_tensor, base_action, episode_length
                    )
                    action_source = "faith"
                    faith_actions_this_episode += 1
                    self.faith_action_count += 1
                else:
                    action = base_action
            # OPTION 2: Planning action (20% by default)
            elif self.use_planning and random.random() < self.planning_freq and len(self.replay_buffer) > 1000:
                action, _ = self.planner.plan_action(obs_with_context)
                action_source = "planning"
                planning_actions_this_episode += 1
                self.planning_count += 1
            # OPTION 3: Reactive Q-learning
            else:
                action = self.policy_net.get_action(obs_with_context, epsilon=epsilon)
                reactive_actions_this_episode += 1
                self.reactive_count += 1

            # Q-Head Analysis (sample periodically)
            if episode_length % 10 == 0:
                self.q_head_analyzer.analyze_step(self.policy_net, obs_with_context, context)

            # Execute action
            next_obs, env_reward, done, info = env.step(action)
            next_obs_with_context = add_context_to_observation(next_obs, context_vector)

            # Extract distances for reward calculation
            pellet_dist, enemy_dist = self.extract_distances_from_obs(next_obs)

            # Enhanced reward calculation
            enhanced_reward, reward_breakdown = reward_system.calculate_reward(
                env_reward, info, pellet_dist, enemy_dist
            )

            # REVOLUTION 2: Learn from entity interactions
            if detected_entities:
                for entity in detected_entities:
                    self.entity_world_model.learn_from_interaction(
                        entity_id=entity['type_id'],
                        reward_delta=enhanced_reward,
                        died=info.get('died', False),
                        collected=info.get('collected_reward', False)
                    )

            # REVOLUTION 4: Record for mechanic detection
            event = None
            if info.get('collected_reward', False):
                event = {'type': 'reward_collected'}

            self.mechanic_detectors[context].observe(
                reward=enhanced_reward,
                action=action,
                event=event,
                entity_distance=enemy_dist
            )

            # Track faith discoveries (novelty)
            if action_source == "faith":
                self.current_episode_faith_reward += enhanced_reward

                # Detect novel discoveries (unusually high reward from faith)
                if enhanced_reward > 50:  # Threshold for "novel discovery"
                    self.faith_discoveries.append({
                        'episode': len(self.episode_rewards),
                        'step': episode_length,
                        'reward': enhanced_reward,
                        'pattern_id': self.active_faith_pattern.pattern_id
                    })

            # Level completion logic
            if info.get('rewards_left', 0) == 0 and not level_completed:
                level_completed = True
                current_level = self.context_levels[context]
                completion_bonus = level_config['completion_bonus']
                enhanced_reward += completion_bonus
                reward_breakdown['level'] = completion_bonus

                # Advance level
                self.context_levels[context] = min(5, current_level + 1)
                self.level_completions[context] += 1

                # Spawn new level
                new_level_config = reward_system.get_current_level_config(self.context_levels[context])
                env = ExpandedTemporalRandom2DEnv(  # EXPANDED: Use expanded observer environment
                    grid_size=(self.env_size, self.env_size),
                    num_entities=new_level_config['enemies'],
                    num_rewards=new_level_config['pellets']
                )
                next_obs = env.reset()
                next_obs_with_context = add_context_to_observation(next_obs, context_vector)
                done = False
                level_completed = False

            # Store transition
            transition = {
                'state': obs_with_context.copy(),
                'action': action,
                'reward': enhanced_reward,
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

                # Train entity discovery model
                entity_loss = self._train_entity_discovery_step()

            # Update target network
            if self.steps_done % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            obs_with_context = next_obs_with_context
            episode_reward += env_reward
            episode_enhanced_reward += enhanced_reward
            episode_length += 1
            self.steps_done += 1

        # END OF EPISODE: Update faith pattern fitness
        if self.active_faith_pattern:
            # Calculate novelty score (did faith discover something new?)
            novelty_score = 0.0
            if self.current_episode_faith_reward > self.baseline_reward:
                novelty_score = (self.current_episode_faith_reward - self.baseline_reward) / 100.0

            # Update pattern fitness
            reward_delta = episode_enhanced_reward - np.mean(self.episode_rewards[-10:]) if self.episode_rewards else episode_enhanced_reward
            self.active_faith_pattern.update_fitness(reward_delta, novelty_score)

        self.baseline_reward = episode_enhanced_reward

        return episode_enhanced_reward, episode_length

    def _train_entity_discovery_step(self):
        """Train entity discovery world model"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        indices = np.random.choice(len(self.replay_buffer.buffer), self.batch_size, replace=False)
        transitions = [self.replay_buffer.buffer[i] for i in indices]

        states = torch.FloatTensor(np.stack([t['state'] for t in transitions]))
        actions = torch.LongTensor(np.array([t['action'] for t in transitions]))
        rewards = torch.FloatTensor(np.array([t['reward'] for t in transitions]))
        next_states = torch.FloatTensor(np.stack([t['next_state'] for t in transitions]))
        dones = torch.FloatTensor(np.array([t['done'] for t in transitions]))

        # Forward pass
        pred_next_states, pred_rewards, pred_dones = self.entity_world_model(states, actions)

        # Compute losses
        state_loss = torch.nn.functional.mse_loss(pred_next_states, next_states)
        reward_loss = torch.nn.functional.mse_loss(pred_rewards.squeeze(), rewards)
        done_loss = torch.nn.functional.binary_cross_entropy(pred_dones.squeeze(), dones)

        total_loss = state_loss + reward_loss + done_loss

        # Optimize
        self.entity_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.entity_world_model.parameters(), 1.0)
        self.entity_optimizer.step()

        return total_loss.item()

    def train(self, num_episodes, log_every=100):
        """
        Main training loop with evolutionary updates.

        ENHANCEMENT: Adds faith evolution and discovery logging.
        """
        print("=" * 70)
        print("STARTING FAITH-BASED EVOLUTIONARY TRAINING")
        print("=" * 70)
        print()

        start_episode = len(self.episode_rewards)
        total_episodes = start_episode + num_episodes

        best_avg_reward = -float('inf')

        for episode in range(start_episode, total_episodes):
            context = self.sample_context()
            self.context_episode_counts[context] += 1

            # Epsilon decay
            epsilon = max(0.01, 1.0 - episode / (total_episodes * 0.5))

            # Train episode
            reward, length = self.train_episode(context, epsilon)

            # Track stats
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            self.context_avg_rewards[context].append(reward)

            # REVOLUTION 1: Evolve faith population periodically
            if (episode + 1) % self.evolution_freq == 0:
                evolution_stats = self.faith_population.evolve(self.episode_rewards)

                print(f"\n{'='*70}")
                print(f"FAITH EVOLUTION - Generation {evolution_stats['generation']}")
                print(f"{'='*70}")
                print(f"  Best Fitness: {evolution_stats['best_fitness']:.2f}")
                print(f"  Diversity: {evolution_stats['diversity']:.4f}")
                print(f"  Mutation Rate: {evolution_stats['mutation_rate']:.3f}")

                pop_stats = self.faith_population.get_statistics()
                print(f"  Behavior Distribution: {pop_stats['behavior_distribution']}")
                print(f"  Total Discoveries: {pop_stats['total_discoveries']}")
                print(f"  Faith Actions: {self.faith_action_count}")
                print()

            # REVOLUTION 3 & 4: Pattern and mechanic detection (every 100 episodes)
            if (episode + 1) % 100 == 0:
                for ctx in ['snake', 'balanced', 'survival']:
                    # Extract universal patterns
                    patterns = self.pattern_extractors[ctx].extract_patterns()
                    if patterns:
                        self.discovered_patterns[ctx] = patterns

                    # Test mechanic hypotheses
                    mechanics = self.mechanic_detectors[ctx].test_hypotheses()
                    if mechanics:
                        self.confirmed_mechanics[ctx] = mechanics

            # Logging
            if (episode + 1) % log_every == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                avg_policy_loss = np.mean(self.policy_losses[-100:]) if self.policy_losses else 0
                avg_wm_loss = np.mean(self.world_model_losses[-100:]) if self.world_model_losses else 0

                print(f"\n{'='*70}")
                print(f"Episode {episode+1}/{total_episodes}")
                print(f"{'='*70}")
                print(f"  Avg Reward (100): {avg_reward:.2f}")
                print(f"  Avg Length (100): {avg_length:.1f}")
                print(f"  Policy Loss: {avg_policy_loss:.4f}")
                print(f"  World Model Loss: {avg_wm_loss:.4f}")
                print(f"  Epsilon: {epsilon:.3f}")
                print(f"  Buffer Size: {len(self.replay_buffer)}")

                # Action distribution
                total_actions = self.planning_count + self.reactive_count + self.faith_action_count
                if total_actions > 0:
                    print(f"\n  Action Distribution:")
                    print(f"    Faith:    {self.faith_action_count/total_actions*100:5.1f}% ({self.faith_action_count})")
                    print(f"    Planning: {self.planning_count/total_actions*100:5.1f}% ({self.planning_count})")
                    print(f"    Reactive: {self.reactive_count/total_actions*100:5.1f}% ({self.reactive_count})")

                # Faith discoveries
                if self.faith_discoveries:
                    recent_discoveries = [d for d in self.faith_discoveries if d['episode'] >= episode - 100]
                    print(f"\n  Faith Discoveries (last 100 episodes): {len(recent_discoveries)}")
                    if recent_discoveries:
                        avg_discovery_reward = np.mean([d['reward'] for d in recent_discoveries])
                        print(f"    Avg discovery reward: {avg_discovery_reward:.2f}")

                # REVOLUTION 2: Entity discoveries
                print(f"\n  Entity Discoveries:")
                entity_summary = self.entity_world_model.get_discovery_summary()
                print(f"    Total entity types: {entity_summary['total_discovered']}")
                for entity_id, info in list(entity_summary['entities'].items())[:5]:
                    print(f"      Entity #{entity_id}: {info['type']} ({info['interactions']} interactions)")

                # REVOLUTION 3: Universal patterns
                print(f"\n  Universal Patterns Discovered:")
                for ctx in ['snake', 'balanced', 'survival']:
                    if self.discovered_patterns[ctx]:
                        print(f"    {ctx}: {list(self.discovered_patterns[ctx].keys())}")

                # REVOLUTION 4: Confirmed mechanics
                print(f"\n  Hidden Mechanics Confirmed:")
                for ctx in ['snake', 'balanced', 'survival']:
                    mechanics = self.mechanic_detectors[ctx].get_active_mechanics()
                    if mechanics:
                        print(f"    {ctx}:")
                        for name, desc in mechanics.items():
                            print(f"      - {desc}")

                print()

                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.save(f"checkpoints/faith_evolution_{timestamp}_best")
                    print(f"  [BEST] Saved model (avg reward: {avg_reward:.2f})")
                    print()

        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total episodes: {len(self.episode_rewards)}")
        print(f"Best avg reward: {best_avg_reward:.2f}")
        print(f"Faith discoveries: {len(self.faith_discoveries)}")

        # Final summaries
        print(f"\nFINAL ENTITY DISCOVERIES:")
        entity_summary = self.entity_world_model.get_discovery_summary()
        for entity_id, info in entity_summary['entities'].items():
            print(f"  Entity #{entity_id}: {info['type']}")

        print(f"\nFINAL UNIVERSAL PATTERNS:")
        for ctx in ['snake', 'balanced', 'survival']:
            if self.discovered_patterns[ctx]:
                print(f"  {ctx}: {list(self.discovered_patterns[ctx].keys())}")

        print(f"\nFINAL HIDDEN MECHANICS:")
        for ctx in ['snake', 'balanced', 'survival']:
            mechanics = self.mechanic_detectors[ctx].get_active_mechanics()
            if mechanics:
                print(f"  {ctx}:")
                for name, desc in mechanics.items():
                    print(f"    - {desc}")

    def save(self, base_path):
        """Save models WITH faith system data"""
        # Call parent save first
        super().save(base_path)

        # Load the policy checkpoint and add faith data
        policy_path = f"{base_path}_policy.pth"
        checkpoint = torch.load(policy_path, map_location='cpu', weights_only=False)

        # Add faith system data
        checkpoint['faith_count'] = self.faith_action_count
        checkpoint['faith_discoveries'] = self.faith_discoveries
        checkpoint['faith_discovery_count'] = len(self.faith_discoveries)

        # Save faith population
        checkpoint['faith_population'] = [
            {
                'pattern_id': p.pattern_id,
                'fitness': p.fitness,
                'age': p.age,
                'discoveries': p.discoveries,
                'behavior_types': p.behavior_types
            }
            for p in self.faith_population.patterns
        ]
        checkpoint['faith_population_size'] = len(self.faith_population.patterns)
        checkpoint['faith_generation'] = self.faith_population.generation  # Save generation counter

        # Entity discovery stats
        checkpoint['entity_types_discovered'] = len(self.entity_world_model.discovered_entities)
        checkpoint['entity_world_model'] = self.entity_world_model.state_dict()

        # Pattern detection stats
        total_patterns = sum(len(patterns) for patterns in self.discovered_patterns.values())
        checkpoint['patterns_detected'] = total_patterns
        checkpoint['discovered_patterns'] = self.discovered_patterns

        # Mechanic detection stats
        total_mechanics = sum(len(mechs) for mechs in self.confirmed_mechanics.values())
        checkpoint['mechanics_confirmed'] = total_mechanics
        checkpoint['confirmed_mechanics'] = self.confirmed_mechanics

        # Re-save with faith data
        torch.save(checkpoint, policy_path)
        print(f"  [FAITH DATA] Saved: {self.faith_action_count} actions, {len(self.faith_discoveries)} discoveries, {len(self.entity_world_model.discovered_entities)} entities")

    def load(self, policy_path, world_model_path=None):
        """Load checkpoint WITH faith system data restoration"""
        # Call parent load first
        super().load(policy_path, world_model_path)

        # Load faith system data
        checkpoint = torch.load(policy_path, map_location='cpu', weights_only=False)

        # Restore faith counters
        self.faith_action_count = checkpoint.get('faith_count', 0)
        self.faith_discoveries = checkpoint.get('faith_discoveries', [])

        # Restore faith population
        if 'faith_population' in checkpoint:
            for i, pattern_data in enumerate(checkpoint['faith_population']):
                if i < len(self.faith_population.patterns):
                    self.faith_population.patterns[i].fitness = pattern_data.get('fitness', 0)
                    self.faith_population.patterns[i].age = pattern_data.get('age', 0)
                    self.faith_population.patterns[i].discoveries = pattern_data.get('discoveries', 0)
                    # Restore behavior_types if available
                    if 'behavior_types' in pattern_data:
                        self.faith_population.patterns[i].behavior_types = pattern_data['behavior_types']

            # Restore generation counter
            self.faith_population.generation = checkpoint.get('faith_generation', 0)
            print(f"  [FAITH] Restored generation: {self.faith_population.generation}")

        # Restore entity world model
        if 'entity_world_model' in checkpoint:
            self.entity_world_model.load_state_dict(checkpoint['entity_world_model'])

        # Restore patterns and mechanics
        if 'discovered_patterns' in checkpoint:
            self.discovered_patterns = checkpoint['discovered_patterns']
        if 'confirmed_mechanics' in checkpoint:
            self.confirmed_mechanics = checkpoint['confirmed_mechanics']

        print(f"  [FAITH DATA] Restored: {self.faith_action_count} actions, {len(self.faith_discoveries)} discoveries")
        if 'entity_types_discovered' in checkpoint:
            print(f"  [ENTITIES] Restored: {checkpoint['entity_types_discovered']} entity types")


def main():
    parser = argparse.ArgumentParser(description='Train Faith-Based Evolutionary Agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--log-every', type=int, default=100, help='Log frequency')
    parser.add_argument('--env-size', type=int, default=20, help='Environment size')
    parser.add_argument('--num-rewards', type=int, default=10, help='Number of rewards')
    parser.add_argument('--use-planning', action='store_true', default=True, help='Enable world model planning')
    parser.add_argument('--planning-freq', type=float, default=0.2, help='Planning frequency (0-1)')
    parser.add_argument('--planning-horizon', type=int, default=20, help='Planning lookahead steps (EXPANDED: 20 vs 5)')
    parser.add_argument('--faith-freq', type=float, default=0.3, help='Faith action frequency (0-1)')
    parser.add_argument('--faith-population', type=int, default=20, help='Faith pattern population size')
    parser.add_argument('--evolution-freq', type=int, default=50, help='Evolve faith patterns every N episodes')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    print("=" * 70)
    print("FAITH-BASED EVOLUTIONARY AGENT TRAINING")
    print("=" * 70)
    print(f"Episodes: {args.episodes}")
    print(f"Faith frequency: {args.faith_freq*100:.0f}%")
    print(f"Faith population: {args.faith_population}")
    print(f"Evolution frequency: Every {args.evolution_freq} episodes")
    print(f"Planning: {'ENABLED' if args.use_planning else 'DISABLED'} ({args.planning_freq*100:.0f}%)")
    print()

    # Create trainer
    trainer = FaithBasedEvolutionaryTrainer(
        env_size=args.env_size,
        num_rewards=args.num_rewards,
        use_planning=args.use_planning,
        planning_freq=args.planning_freq,
        planning_horizon=args.planning_horizon,
        faith_freq=args.faith_freq,
        faith_population_size=args.faith_population,
        evolution_freq=args.evolution_freq
    )

    # Load checkpoint if resuming
    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume}")
        print()

    # Train
    trainer.train(num_episodes=args.episodes, log_every=args.log_every)

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save(f"checkpoints/faith_evolution_{timestamp}_final")


if __name__ == '__main__':
    main()

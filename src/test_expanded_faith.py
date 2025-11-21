"""
Comprehensive Test Suite for EXPANDED Faith-Based Evolutionary Agent

Tests all 4 revolutionary components with EXPANDED spatial-temporal capacity:
1. Faith Pattern Evolution - Which patterns discovered novel strategies?
2. Entity Discovery - What entities learned? Correct classification?
3. Universal Pattern Transfer - Do patterns transfer across games?
4. Mechanic Detection - Which hidden rules were confirmed?

EXPANDED OBSERVER FEATURES:
- 16 rays × 15 tiles (2x spatial coverage vs baseline)
- Multi-scale temporal: Micro (5) + Meso (20) + Macro (50) frames
- Ghost behavior mode detection (chase/scatter/random)
- 180-dim observations (vs 92 baseline) + 3 context = 183 total
- 20-step planning horizon (vs 5 baseline)

MAINTAINS COMPARABILITY:
- Same games (Snake, Pac-Man, Dungeon)
- Same metrics (scores, steps, context distribution)
- Same episode counts

ADDS REVOLUTIONARY METRICS:
- Faith discovery rate and effectiveness
- Entity classification accuracy
- Pattern transfer success rate
- Mechanic confirmation confidence
- Action distribution (faith/planning/reactive)

Usage:
    python test_expanded_faith.py checkpoints/expanded_faith_20251119_201815_best.pth
    python test_expanded_faith.py <model_path> --episodes 100
    python test_expanded_faith.py <model_path> --game pacman --analyze-faith
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

import torch
import numpy as np
import argparse
from collections import defaultdict

# Base system with EXPANDED observer
from context_aware_agent import (
    ContextAwareDQN,
    infer_context_from_observation,
    add_context_to_observation
)
from core.expanded_temporal_observer import ExpandedTemporalObserver  # EXPANDED!
from core.planning_test_games import SnakeGame, PacManGame, DungeonGame

# Revolutionary modules
from core.faith_system import FaithPattern, FaithPopulation
from core.entity_discovery import EntityDiscoveryWorldModel, EntityBehaviorLearner
from core.pattern_transfer import UniversalPatternExtractor
from core.mechanic_detectors import MechanicDetector


class FaithTestOracle:
    """
    Oracle for comprehensive faith-based system testing.

    Tracks all 4 revolutionary components during testing:
    - Faith patterns and their discoveries
    - Entity types and classifications
    - Universal patterns detected
    - Hidden mechanics confirmed
    """

    def __init__(self, entity_world_model, faith_population=None):
        self.entity_world_model = entity_world_model
        self.faith_population = faith_population

        # Pattern extractor for current game
        self.pattern_extractor = UniversalPatternExtractor()

        # Mechanic detector for current game
        self.mechanic_detector = MechanicDetector()

        # Entity learner
        self.entity_learner = EntityBehaviorLearner()

        # Tracking
        self.faith_discoveries = []
        self.entity_detections_timeline = []
        self.pattern_detections = {}
        self.mechanic_confirmations = {}

        # Action tracking
        self.faith_actions = 0
        self.planning_actions = 0
        self.reactive_actions = 0

        # Performance by action type
        self.faith_action_rewards = []
        self.planning_action_rewards = []
        self.reactive_action_rewards = []

    def observe_step(self, obs, action, reward, action_source='reactive',
                     entity_distance=None, step=None):
        """
        Record step for comprehensive analysis.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            action_source: 'faith', 'planning', or 'reactive'
            entity_distance: Distance to nearest entity
            step: Current step number
        """
        # Track action distribution
        if action_source == 'faith':
            self.faith_actions += 1
            self.faith_action_rewards.append(reward)

            # Track faith discoveries (unusually high reward)
            if reward > 50:
                self.faith_discoveries.append({
                    'step': step,
                    'reward': reward,
                    'action': action
                })

        elif action_source == 'planning':
            self.planning_actions += 1
            self.planning_action_rewards.append(reward)
        else:
            self.reactive_actions += 1
            self.reactive_action_rewards.append(reward)

        # Detect entities
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        entity_probs, detected_entities = self.entity_world_model.detect_entities(obs_tensor)

        if detected_entities:
            self.entity_detections_timeline.append({
                'step': step,
                'entities': detected_entities
            })

            # Record for pattern extraction
            self.pattern_extractor.observe(detected_entities, obs)

            # Record for entity learning
            self.entity_learner.observe(step, detected_entities)

        # Record for mechanic detection
        event = None
        if reward > 0:
            event = {'type': 'reward_collected'}

        self.mechanic_detector.observe(
            reward=reward,
            action=action,
            event=event,
            entity_distance=entity_distance
        )

    def analyze_patterns(self):
        """Extract and analyze universal patterns detected"""
        self.pattern_detections = self.pattern_extractor.extract_patterns()
        return self.pattern_detections

    def analyze_mechanics(self):
        """Test mechanic hypotheses"""
        self.mechanic_confirmations = self.mechanic_detector.test_hypotheses()
        return self.mechanic_confirmations

    def get_entity_summary(self):
        """Get entity discovery summary"""
        return self.entity_world_model.get_discovery_summary()

    def get_comprehensive_report(self):
        """Generate comprehensive test report"""
        # Action distribution
        total_actions = self.faith_actions + self.planning_actions + self.reactive_actions

        report = {
            'action_distribution': {
                'faith': {
                    'count': self.faith_actions,
                    'percent': (self.faith_actions / total_actions * 100) if total_actions > 0 else 0,
                    'avg_reward': np.mean(self.faith_action_rewards) if self.faith_action_rewards else 0
                },
                'planning': {
                    'count': self.planning_actions,
                    'percent': (self.planning_actions / total_actions * 100) if total_actions > 0 else 0,
                    'avg_reward': np.mean(self.planning_action_rewards) if self.planning_action_rewards else 0
                },
                'reactive': {
                    'count': self.reactive_actions,
                    'percent': (self.reactive_actions / total_actions * 100) if total_actions > 0 else 0,
                    'avg_reward': np.mean(self.reactive_action_rewards) if self.reactive_action_rewards else 0
                }
            },
            'faith_discoveries': {
                'count': len(self.faith_discoveries),
                'discoveries': self.faith_discoveries
            },
            'entity_summary': self.get_entity_summary(),
            'patterns': self.pattern_detections,
            'mechanics': {
                name: mechanic.description
                for name, mechanic in self.mechanic_confirmations.items()
            },
            'pattern_stats': self.pattern_extractor.get_statistics(),
            'mechanic_stats': self.mechanic_detector.get_statistics()
        }

        return report


def _plan_action(agent, world_model, state, planning_horizon=20):  # EXPANDED: 20 vs 5
    """Use world model to plan best action via lookahead"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    best_action = None
    best_return = -float('inf')

    # Try each action
    for action in range(4):
        total_return = 0.0

        # Monte Carlo: simulate multiple rollouts
        num_rollouts = 5
        for _ in range(num_rollouts):
            rollout_return = _simulate_rollout(agent, world_model, state_tensor, action, planning_horizon)
            total_return += rollout_return

        avg_return = total_return / num_rollouts

        if avg_return > best_return:
            best_return = avg_return
            best_action = action

    return best_action


def _simulate_rollout(agent, world_model, state, first_action, planning_horizon=20):  # EXPANDED: 20 vs 5
    """Simulate one trajectory using world model"""
    current_state = state.clone()
    total_return = 0.0
    discount = 1.0
    gamma = 0.99

    with torch.no_grad():
        # Take first action
        action_tensor = torch.LongTensor([first_action])
        next_state, reward, done = world_model(current_state, action_tensor)
        total_return += reward.item() * discount
        discount *= gamma

        if done.item() > 0.5:
            return total_return

        current_state = next_state

        # Simulate remaining horizon steps using policy
        for _ in range(planning_horizon - 1):
            q_values = agent.get_combined_q(current_state)
            action = q_values.argmax(dim=1).item()

            action_tensor = torch.LongTensor([action])
            next_state, reward, done = world_model(current_state, action_tensor)

            total_return += reward.item() * discount
            discount *= gamma

            if done.item() > 0.5:
                break

            current_state = next_state

    return total_return


def test_game_with_faith(agent, observer, game, game_name, num_episodes=50,
                         world_model=None, entity_world_model=None,
                         faith_population=None,
                         planning_freq=0.2, planning_horizon=20,  # EXPANDED: 20 vs 5
                         faith_freq=0.3, analyze_faith=True):
    """
    Test agent with comprehensive faith-based analysis.

    Args:
        agent: Policy network
        observer: EXPANDED temporal observer (180 dims)
        game: Game instance
        game_name: Game name
        num_episodes: Number of test episodes
        world_model: World model for planning
        entity_world_model: Entity discovery model
        faith_population: Faith pattern population
        planning_freq: Planning action frequency
        planning_horizon: Planning lookahead steps (EXPANDED: 20 vs 5)
        faith_freq: Faith action frequency
        analyze_faith: Whether to perform deep faith analysis
    """
    # Initialize oracle
    oracle = FaithTestOracle(entity_world_model, faith_population) if analyze_faith else None

    # Standard metrics
    scores = []
    steps_list = []
    context_counts = {'snake': 0, 'balanced': 0, 'survival': 0}

    for episode in range(num_episodes):
        game.reset()
        observer.reset()
        game_state = game._get_game_state()

        # Select faith pattern for this episode (if available)
        active_faith_pattern = None
        if faith_population and analyze_faith:
            active_faith_pattern = faith_population.select_pattern_for_episode()

        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 1000:
            # Get observation (EXPANDED: 180 dims!)
            obs = observer.observe(game_state)

            # Infer context
            context_vector = infer_context_from_observation(obs)

            # Track context distribution
            if context_vector[0] == 1.0:
                context_counts['snake'] += 1
            elif context_vector[1] == 1.0:
                context_counts['balanced'] += 1
            else:
                context_counts['survival'] += 1

            # Add context to observation (183 dims total)
            obs_with_context = add_context_to_observation(obs, context_vector)

            # ACTION SELECTION: Faith → Planning → Reactive
            action_source = 'reactive'

            # Option 1: Faith-based action
            if active_faith_pattern and np.random.random() < faith_freq:
                obs_tensor = torch.FloatTensor(obs_with_context).unsqueeze(0)
                base_action = agent.get_action(obs_with_context, epsilon=0.0)

                if active_faith_pattern.should_override(steps, obs_tensor):
                    action = active_faith_pattern.get_action(obs_tensor, base_action, steps)
                    action_source = 'faith'
                else:
                    action = base_action

            # Option 2: Planning action (EXPANDED: 20-step horizon!)
            elif world_model is not None and np.random.random() < planning_freq:
                action = _plan_action(agent, world_model, obs_with_context, planning_horizon)
                action_source = 'planning'

            # Option 3: Reactive action
            else:
                action = agent.get_action(obs_with_context, epsilon=0.0)

            # Execute action
            game_state, reward, done = game.step(action)
            total_reward += reward

            # Record for analysis
            if oracle:
                # Extract entity distance from observation (adjusted for 180 dims)
                entity_distance = obs[68] if len(obs) > 68 else None  # EXPANDED: adjusted index

                oracle.observe_step(
                    obs=obs_with_context,
                    action=action,
                    reward=reward,
                    action_source=action_source,
                    entity_distance=entity_distance,
                    step=steps
                )

            steps += 1

        scores.append(game_state['score'])
        steps_list.append(steps)

    # Analyze patterns and mechanics
    if oracle:
        oracle.analyze_patterns()
        oracle.analyze_mechanics()

    # Print standard results
    print(f"\n{'='*70}")
    print(f"{game_name.upper()} TEST RESULTS (EXPANDED OBSERVER)")
    print(f"{'='*70}")
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")
    print(f"Average Steps: {np.mean(steps_list):.1f}")
    print()

    print("Context Distribution:")
    total_steps = sum(context_counts.values())
    for context, count in context_counts.items():
        pct = (count / total_steps * 100) if total_steps > 0 else 0
        print(f"  {context:8s}: {count:5d} steps ({pct:5.1f}%)")

    # Print revolutionary metrics
    if oracle:
        report = oracle.get_comprehensive_report()

        print(f"\n{'-'*70}")
        print("REVOLUTIONARY METRICS ANALYSIS")
        print(f"{'-'*70}")

        # 1. Action Distribution & Effectiveness
        print("\n[1] ACTION DISTRIBUTION & EFFECTIVENESS:")
        action_dist = report['action_distribution']
        for action_type in ['faith', 'planning', 'reactive']:
            info = action_dist[action_type]
            print(f"  {action_type.capitalize():10s}: {info['percent']:5.1f}% "
                  f"({info['count']:4d} actions) | Avg reward: {info['avg_reward']:6.2f}")

        # Faith effectiveness
        if action_dist['faith']['count'] > 0 and action_dist['reactive']['count'] > 0:
            faith_advantage = action_dist['faith']['avg_reward'] - action_dist['reactive']['avg_reward']
            if faith_advantage > 0:
                print(f"\n  Faith Advantage: +{faith_advantage:.2f} reward per action (DISCOVERING!)")
            else:
                print(f"\n  Faith Advantage: {faith_advantage:.2f} (exploring, not yet optimizing)")

        # 2. Faith Discoveries
        print(f"\n[2] FAITH DISCOVERIES:")
        discoveries = report['faith_discoveries']
        print(f"  Total novel discoveries: {discoveries['count']}")
        if discoveries['discoveries']:
            print(f"  Discovery rewards: {[d['reward'] for d in discoveries['discoveries'][:5]]}")
            print(f"  Discovery steps: {[d['step'] for d in discoveries['discoveries'][:5]]}")

        # 3. Entity Classification
        print(f"\n[3] ENTITY DISCOVERY & CLASSIFICATION:")
        entity_summary = report['entity_summary']
        print(f"  Total entity types discovered: {entity_summary['total_discovered']}")

        # Group by type
        entity_type_counts = defaultdict(int)
        for entity_id, info in entity_summary['entities'].items():
            entity_type_counts[info['type']] += 1

        print(f"  Entity classification breakdown:")
        for entity_type, count in sorted(entity_type_counts.items()):
            print(f"    {entity_type:20s}: {count:2d} instances")

        # Show top 3 most interacted entities
        if entity_summary['entities']:
            print(f"\n  Top interacted entities:")
            sorted_entities = sorted(
                entity_summary['entities'].items(),
                key=lambda x: x[1]['interactions'],
                reverse=True
            )
            for entity_id, info in sorted_entities[:3]:
                print(f"    Entity #{entity_id}: {info['type']:20s} "
                      f"({info['interactions']:3d} interactions, "
                      f"reward: {info['avg_reward']:6.2f})")

        # 4. Universal Patterns
        print(f"\n[4] UNIVERSAL PATTERN DETECTION:")
        patterns = report['patterns']
        if patterns:
            print(f"  Detected {len(patterns)} universal patterns:")
            for pattern_name, strategy in patterns.items():
                print(f"    {pattern_name.upper().replace('_', ' ')}:")
                print(f"      Confidence: {strategy['confidence']:.2f}")
                print(f"      Priority: {strategy['priority']}")
        else:
            print(f"  No universal patterns detected (need more episodes)")

        # 5. Hidden Mechanics
        print(f"\n[5] HIDDEN MECHANIC CONFIRMATION:")
        mechanics = report['mechanics']
        if mechanics:
            print(f"  Confirmed {len(mechanics)} hidden mechanics:")
            for mechanic_name, description in mechanics.items():
                print(f"    {mechanic_name.upper().replace('_', ' ')}:")
                print(f"      {description}")
        else:
            print(f"  No hidden mechanics confirmed (need more episodes)")

        # 6. Statistics summary
        print(f"\n[6] DETECTION STATISTICS:")
        pattern_stats = report['pattern_stats']
        mechanic_stats = report['mechanic_stats']
        print(f"  Pattern observations: {pattern_stats['total_observations']}")
        print(f"  Patterns detected: {pattern_stats['patterns_detected']}")
        print(f"  Mechanic observations: {mechanic_stats['total_observations']}")
        print(f"  Mechanics confirmed: {mechanic_stats['mechanics_confirmed']}")

    return {
        'scores': scores,
        'steps': steps_list,
        'avg_score': np.mean(scores),
        'context_counts': context_counts,
        'faith_report': oracle.get_comprehensive_report() if oracle else None
    }


def main():
    parser = argparse.ArgumentParser(description='Comprehensive EXPANDED Faith-Based Agent Test')
    parser.add_argument('model_path', help='Path to policy checkpoint')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per game')
    parser.add_argument('--game', choices=['snake', 'pacman', 'dungeon', 'all'], default='all')
    parser.add_argument('--no-planning', action='store_true', help='Disable planning')
    parser.add_argument('--planning-freq', type=float, default=0.2, help='Planning frequency')
    parser.add_argument('--planning-horizon', type=int, default=20, help='Planning horizon (EXPANDED: 20 vs 5)')
    parser.add_argument('--no-faith', action='store_true', help='Disable faith actions')
    parser.add_argument('--faith-freq', type=float, default=0.3, help='Faith action frequency')
    parser.add_argument('--analyze-faith', action='store_true', default=True,
                        help='Deep faith analysis (default: enabled)')
    parser.add_argument('--simple-test', action='store_true',
                        help='Simple test without faith analysis')

    args = parser.parse_args()

    analyze_faith = args.analyze_faith and not args.simple_test

    # Load agent
    print(f"{'='*70}")
    print(f"EXPANDED FAITH-BASED EVOLUTIONARY AGENT TEST")
    print(f"{'='*70}")
    print(f"Model: {args.model_path}")
    print(f"Episodes per game: {args.episodes}")
    print(f"Faith analysis: {'ENABLED' if analyze_faith else 'DISABLED'}")
    print(f"Observer: EXPANDED (180 dims + 3 context = 183 total)")
    print(f"Planning horizon: {args.planning_horizon} steps")
    print()

    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)

    # Load policy network with EXPANDED dimensions
    agent = ContextAwareDQN(obs_dim=183, action_dim=4)  # EXPANDED: 183 vs 95
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    # Display training info
    print("TRAINING INFORMATION:")
    print(f"  Episodes trained: {len(checkpoint.get('episode_rewards', []))}")
    print(f"  Steps: {checkpoint.get('steps_done', 0)}")

    # Display action counts
    faith_count = checkpoint.get('faith_action_count', 0)
    planning_count = checkpoint.get('planning_count', 0)
    reactive_count = checkpoint.get('reactive_count', 0)
    total_actions = faith_count + planning_count + reactive_count

    if total_actions > 0:
        print(f"\n  Training action distribution:")
        print(f"    Faith:    {faith_count/total_actions*100:5.1f}% ({faith_count:,})")
        print(f"    Planning: {planning_count/total_actions*100:5.1f}% ({planning_count:,})")
        print(f"    Reactive: {reactive_count/total_actions*100:5.1f}% ({reactive_count:,})")

    # Display faith discoveries
    faith_discoveries = checkpoint.get('faith_discoveries', [])
    if faith_discoveries:
        print(f"\n  Faith discoveries during training: {len(faith_discoveries)}")
        print(f"    Avg discovery reward: {np.mean([d['reward'] for d in faith_discoveries]):.2f}")

    # Display context distribution
    if 'context_episode_counts' in checkpoint:
        print(f"\n  Training context distribution:")
        for ctx, count in checkpoint['context_episode_counts'].items():
            print(f"    {ctx:8s}: {count} episodes")

    # Display level progression
    if 'context_levels' in checkpoint:
        print(f"\n  Level progression:")
        for ctx in ['snake', 'balanced', 'survival']:
            level = checkpoint['context_levels'].get(ctx, 1)
            completions = checkpoint.get('level_completions', {}).get(ctx, 0)
            print(f"    {ctx:8s}: Level {level} ({completions} completions)")

    print()

    # Load world model for planning
    world_model = None
    if not args.no_planning:
        base_path = args.model_path.replace('_policy.pth', '').replace('_best.pth', '').replace('_final.pth', '')
        world_model_path = f"{base_path}_world_model.pth"

        if os.path.exists(world_model_path):
            print(f"Loading world model: {world_model_path}")

            # Load checkpoint first to detect architecture
            wm_checkpoint = torch.load(world_model_path, map_location='cpu', weights_only=False)
            state_dict = wm_checkpoint['model']

            # Check for FIXED world model marker in policy checkpoint
            policy_checkpoint = checkpoint
            world_model_type = policy_checkpoint.get('world_model_type', 'standard')

            if world_model_type == 'context_aware_fixed':
                # FIXED world model with context-aware architecture
                print(f"  Detected: FIXED context-aware world model")
                from core.context_aware_world_model import ContextAwareWorldModel

                obs_dim = policy_checkpoint.get('world_model_obs_dim', 180)
                context_dim = policy_checkpoint.get('world_model_context_dim', 3)

                # Detect hidden_dim
                if 'obs_predictor.0.weight' in state_dict:
                    hidden_dim = state_dict['obs_predictor.0.weight'].shape[0]
                else:
                    hidden_dim = 256

                world_model = ContextAwareWorldModel(
                    obs_dim=obs_dim,
                    context_dim=context_dim,
                    action_dim=4,
                    hidden_dim=hidden_dim
                )
                world_model.load_state_dict(state_dict)
                world_model.eval()
                print(f"  Context-Aware World Model loaded (obs={obs_dim}, hidden={hidden_dim})")
            else:
                # Old architecture - try to detect
                if 'state_predictor.0.weight' in state_dict:
                    hidden_dim = state_dict['state_predictor.0.weight'].shape[0]
                    print(f"  Detected standard world model: hidden_dim={hidden_dim}")
                else:
                    hidden_dim = 256
                    print(f"  Using default architecture: hidden_dim={hidden_dim}")

                # Try entity discovery world model first
                try:
                    from core.entity_discovery import EntityDiscoveryWorldModel
                    world_model = EntityDiscoveryWorldModel(obs_dim=183, action_dim=4)
                    world_model.load_state_dict(state_dict)
                    world_model.eval()
                    print(f"  Entity Discovery World Model loaded (EXPANDED: 183 dims)")
                except Exception as e:
                    # Fall back to standard world model
                    from core.world_model import WorldModelNetwork
                    world_model = WorldModelNetwork(state_dim=183, action_dim=4, hidden_dim=hidden_dim)
                    world_model.load_state_dict(state_dict)
                    world_model.eval()
                    print(f"  Standard World Model loaded (EXPANDED: 183 dims, hidden={hidden_dim})")

            print(f"  Planning: {args.planning_freq*100:.0f}% frequency, horizon {args.planning_horizon}")
        else:
            print(f"World model not found: {world_model_path}")
            print(f"  Planning DISABLED")

    # Load entity discovery model
    entity_world_model = None
    if analyze_faith:
        # Try to load or create entity discovery model
        try:
            from core.entity_discovery import EntityDiscoveryWorldModel
            entity_world_model = EntityDiscoveryWorldModel(obs_dim=183, action_dim=4)  # EXPANDED: 183 vs 95

            # Try to load trained weights
            base_path = args.model_path.replace('_policy.pth', '').replace('_best.pth', '').replace('_final.pth', '')
            entity_model_path = f"{base_path}_entity_model.pth"

            if os.path.exists(entity_model_path):
                entity_checkpoint = torch.load(entity_model_path, map_location='cpu', weights_only=False)
                entity_world_model.load_state_dict(entity_checkpoint['model'])
                entity_world_model.eval()
                print(f"\nEntity Discovery Model loaded from: {entity_model_path}")
                print(f"  (EXPANDED: 183 dims)")
            else:
                # Use untrained model (will still classify based on observations)
                print(f"\nEntity Discovery Model: Using fresh model (no saved checkpoint)")
                print(f"  (EXPANDED: 183 dims)")

        except Exception as e:
            print(f"\nWarning: Could not load entity discovery model: {e}")
            analyze_faith = False

    # Load faith population
    faith_population = None
    if analyze_faith and not args.no_faith:
        faith_population = FaithPopulation(population_size=20, signature_dim=32)
        print(f"\nFaith Pattern Population: 20 patterns loaded")
        print(f"  Faith frequency: {args.faith_freq*100:.0f}%")

    print()

    # Test parameters - EXPANDED observer!
    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)  # EXPANDED!
    results = {}

    # Test games
    if args.game in ['snake', 'all']:
        print(f"\n{'='*70}")
        print(f"TESTING: SNAKE GAME (EXPANDED OBSERVER)")
        print(f"{'='*70}")
        game = SnakeGame(size=20)
        results['snake'] = test_game_with_faith(
            agent, observer, game, "Snake", args.episodes,
            world_model=world_model,
            entity_world_model=entity_world_model,
            faith_population=faith_population,
            planning_freq=args.planning_freq,
            planning_horizon=args.planning_horizon,
            faith_freq=args.faith_freq,
            analyze_faith=analyze_faith
        )

    if args.game in ['pacman', 'all']:
        print(f"\n{'='*70}")
        print(f"TESTING: PAC-MAN GAME (EXPANDED OBSERVER)")
        print(f"{'='*70}")
        game = PacManGame(size=20)
        results['pacman'] = test_game_with_faith(
            agent, observer, game, "Pac-Man", args.episodes,
            world_model=world_model,
            entity_world_model=entity_world_model,
            faith_population=faith_population,
            planning_freq=args.planning_freq,
            planning_horizon=args.planning_horizon,
            faith_freq=args.faith_freq,
            analyze_faith=analyze_faith
        )

    if args.game in ['dungeon', 'all']:
        print(f"\n{'='*70}")
        print(f"TESTING: DUNGEON GAME (EXPANDED OBSERVER)")
        print(f"{'='*70}")
        game = DungeonGame(size=20)
        results['dungeon'] = test_game_with_faith(
            agent, observer, game, "Dungeon", args.episodes,
            world_model=world_model,
            entity_world_model=entity_world_model,
            faith_population=faith_population,
            planning_freq=args.planning_freq,
            planning_horizon=args.planning_horizon,
            faith_freq=args.faith_freq,
            analyze_faith=analyze_faith
        )

    # Overall summary
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY (EXPANDED OBSERVER)")
        print(f"{'='*70}")

        print("\nGame Performance:")
        for game_name, result in results.items():
            print(f"  {game_name:8s}: {result['avg_score']:6.2f} avg score")

        # Faith analysis summary
        if analyze_faith:
            print(f"\n{'='*70}")
            print("CROSS-GAME FAITH ANALYSIS")
            print(f"{'='*70}")

            # Aggregate faith discoveries
            total_discoveries = sum(
                len(result['faith_report']['faith_discoveries']['discoveries'])
                for result in results.values()
                if result['faith_report']
            )
            print(f"\nTotal faith discoveries across all games: {total_discoveries}")

            # Aggregate entity types
            all_entity_types = set()
            for result in results.values():
                if result['faith_report']:
                    entity_summary = result['faith_report']['entity_summary']
                    for entity_id, info in entity_summary['entities'].items():
                        all_entity_types.add(info['type'])

            print(f"Unique entity types discovered: {len(all_entity_types)}")
            print(f"  Types: {', '.join(sorted(all_entity_types))}")

            # Pattern transfer analysis
            print(f"\nPattern Transfer Analysis:")
            pattern_games = defaultdict(list)
            for game_name, result in results.items():
                if result['faith_report']:
                    for pattern_name in result['faith_report']['patterns'].keys():
                        pattern_games[pattern_name].append(game_name)

            if pattern_games:
                print(f"  Patterns detected in multiple games (TRANSFER SUCCESS):")
                for pattern_name, games in pattern_games.items():
                    if len(games) > 1:
                        print(f"    {pattern_name.replace('_', ' ').title()}: {', '.join(games)}")
            else:
                print(f"  No cross-game patterns detected (need more episodes)")

        # Context adaptation check
        print(f"\n{'='*70}")
        print("CONTEXT ADAPTATION CHECK")
        print(f"{'='*70}")

        for game_name, result in results.items():
            dominant_context = max(result['context_counts'].items(), key=lambda x: x[1])
            total_steps = sum(result['context_counts'].values())
            pct = (dominant_context[1] / total_steps * 100) if total_steps > 0 else 0

            print(f"\n{game_name.capitalize()} game:")
            print(f"  Dominant context: {dominant_context[0]} ({pct:.1f}%)")

            if game_name == 'snake' and dominant_context[0] == 'snake' and pct > 80:
                print(f"    [EXCELLENT] Correctly identifies collection-focused context")
            elif game_name == 'pacman' and dominant_context[0] == 'balanced' and pct > 60:
                print(f"    [GOOD] Correctly identifies balanced threat/reward context")
            elif game_name == 'dungeon' and dominant_context[0] == 'survival' and pct > 70:
                print(f"    [EXCELLENT] Correctly identifies high-threat survival context")
            else:
                print(f"    [INFO] Context distribution varies during gameplay")


if __name__ == '__main__':
    main()

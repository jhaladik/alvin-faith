"""
EXPANDED Faith-Based Training - FIXED WORLD MODEL BOTTLENECK

This is an IMPROVED version of train_expanded_faith.py that fixes the
world model bottleneck issue.

KEY FIX:
- Uses ContextAwareWorldModel instead of WorldModelNetwork
- Only predicts observation (180 dims), NOT context (3 dims)
- Removes bottleneck from predicting constant context values
- Expected: Faster convergence, better planning, clearer gradients

Usage:
    # Train from scratch with fixed model
    python train_expanded_faith_fixed.py --episodes 500

    # Resume from old checkpoint (will upgrade to fixed model)
    python train_expanded_faith_fixed.py --episodes 500 \\
        --resume checkpoints/faith_evolution_20251120_091144_final_policy.pth

Note: When resuming from old checkpoint, world model is recreated with
fixed architecture. Policy network is compatible (same input/output dims).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

import torch
import numpy as np
import argparse
from datetime import datetime

# Import the FIXED world model
from core.context_aware_world_model import ContextAwareWorldModel

# Import base training system
from train_expanded_faith import FaithBasedEvolutionaryTrainer

# Other imports
from context_aware_agent import ContextAwareDQN


class FixedFaithEvolutionaryTrainer(FaithBasedEvolutionaryTrainer):
    """
    Faith-based trainer with FIXED world model bottleneck.

    Changes from parent:
    - Uses ContextAwareWorldModel (predicts 180 dims)
    - Instead of WorldModelNetwork (predicts 183 dims)
    - Removes wasted capacity on context prediction
    """

    def __init__(self, *args, **kwargs):
        # Call parent init first
        super().__init__(*args, **kwargs)

        # Replace world model with FIXED version
        print("\n" + "=" * 70)
        print("REPLACING WORLD MODEL WITH FIXED VERSION")
        print("=" * 70)

        # OLD: WorldModelNetwork(state_dim=183) - predicts 183 dims
        # NEW: ContextAwareWorldModel(obs_dim=180) - predicts 180 dims

        old_params = sum(p.numel() for p in self.world_model.parameters())

        self.world_model = ContextAwareWorldModel(
            obs_dim=180,      # Expanded observer dimension
            context_dim=3,    # Context dimension (snake/balanced/survival)
            action_dim=4,     # Actions (up/down/left/right)
            hidden_dim=256    # Expanded capacity
        )

        new_params = sum(p.numel() for p in self.world_model.parameters())

        print(f"  Old world model: {old_params:,} parameters (predicts 183 dims)")
        print(f"  New world model: {new_params:,} parameters (predicts 180 dims)")
        print(f"  Reduction: {old_params - new_params:,} parameters")
        print()
        print("  BOTTLENECK REMOVED:")
        print("    - No longer predicting constant context values")
        print("    - All capacity for observation dynamics")
        print("    - Cleaner gradient signal")
        print()

        # Recreate optimizer with new world model
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=kwargs.get('lr_world_model', 0.0003)
        )

        # Update planner with new world model
        if self.use_planning:
            from train_context_aware_advanced import WorldModelPlanner
            self.planner = WorldModelPlanner(
                self.policy_net,
                self.world_model,
                gamma=kwargs.get('gamma', 0.99),
                num_rollouts=5,
                horizon=kwargs.get('planning_horizon', 20)
            )
            print("  Planner updated with fixed world model")

    def save(self, base_path):
        """Save with fixed world model marker"""
        # Call parent save
        super().save(base_path)

        # Add marker to indicate fixed world model
        policy_path = f"{base_path}_policy.pth"
        checkpoint = torch.load(policy_path, map_location='cpu', weights_only=False)
        checkpoint['world_model_type'] = 'context_aware_fixed'
        checkpoint['world_model_obs_dim'] = 180
        checkpoint['world_model_context_dim'] = 3
        torch.save(checkpoint, policy_path)

        print(f"  [FIXED WM] Saved with context-aware world model marker")


def main():
    parser = argparse.ArgumentParser(description='Train Faith-Based Agent with FIXED World Model')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--log-every', type=int, default=100, help='Log frequency')
    parser.add_argument('--env-size', type=int, default=20, help='Environment size')
    parser.add_argument('--num-rewards', type=int, default=10, help='Number of rewards')
    parser.add_argument('--use-planning', action='store_true', default=True, help='Enable planning')
    parser.add_argument('--planning-freq', type=float, default=0.2, help='Planning frequency')
    parser.add_argument('--planning-horizon', type=int, default=20, help='Planning horizon')
    parser.add_argument('--faith-freq', type=float, default=0.3, help='Faith frequency')
    parser.add_argument('--faith-population', type=int, default=20, help='Faith population size')
    parser.add_argument('--evolution-freq', type=int, default=50, help='Evolution frequency')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    print("=" * 70)
    print("FAITH-BASED AGENT TRAINING - FIXED WORLD MODEL")
    print("=" * 70)
    print(f"Episodes: {args.episodes}")
    print(f"Faith frequency: {args.faith_freq*100:.0f}%")
    print(f"Planning: {'ENABLED' if args.use_planning else 'DISABLED'} ({args.planning_freq*100:.0f}%)")
    print()
    print("KEY IMPROVEMENT:")
    print("  World model bottleneck FIXED")
    print("  - Predicts 180 dims (obs) instead of 183 dims (obs + context)")
    print("  - Context passed through unchanged")
    print("  - Expected: Faster convergence, better planning")
    print()

    # Create trainer with FIXED world model
    trainer = FixedFaithEvolutionaryTrainer(
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
        print(f"\nResuming from: {args.resume}")
        print("  Note: Policy loaded, world model recreated with fixed architecture")
        print()

        # Load policy checkpoint
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        trainer.policy_net.load_state_dict(checkpoint['policy_net'])
        trainer.target_net.load_state_dict(checkpoint['target_net'])

        # Restore training state
        trainer.episode_rewards = checkpoint.get('episode_rewards', [])
        trainer.episode_lengths = checkpoint.get('episode_lengths', [])
        trainer.steps_done = checkpoint.get('steps_done', 0)
        trainer.context_episode_counts = checkpoint.get('context_episode_counts',
                                                       {'snake': 0, 'balanced': 0, 'survival': 0})
        trainer.context_levels = checkpoint.get('context_levels',
                                               {'snake': 1, 'balanced': 1, 'survival': 1})
        trainer.level_completions = checkpoint.get('level_completions',
                                                   {'snake': 0, 'balanced': 0, 'survival': 0})

        # Restore faith data
        if 'faith_action_count' in checkpoint:
            trainer.faith_action_count = checkpoint['faith_action_count']
            trainer.faith_discoveries = checkpoint.get('faith_discoveries', [])

        # Restore counters
        trainer.planning_count = checkpoint.get('planning_count', 0)
        trainer.reactive_count = checkpoint.get('reactive_count', 0)

        print(f"  Resumed from episode {len(trainer.episode_rewards)}")
        print(f"  Total steps: {trainer.steps_done:,}")
        print()

    # Train
    trainer.train(num_episodes=args.episodes, log_every=args.log_every)

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save(f"checkpoints/faith_fixed_{timestamp}_final")
    print(f"\nSaved final model with FIXED world model")


if __name__ == '__main__':
    main()

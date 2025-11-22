"""
Multi-Game Curriculum Training - Foundation Agent

Train on multiple games with curriculum approach to build true transfer capabilities:

CURRICULUM STRATEGY:
- Phase 1 (0-300):     Snake 70% + Pac-Man 30% (Dense rewards)
- Phase 2 (300-650):   Snake 40% + Pac-Man 30% + Local View 30% (Add medium rewards, large world)
- Phase 3 (650-1000):  Snake 30% + Pac-Man 25% + Local View 25% + Dungeon 20% (Add sparse rewards, maze)

KEY FEATURES:
1. Exploration bonus - Rewards visiting new cells (solves Dungeon/Local View)
2. Multi-game sampling - Prevents overfitting to single game
3. Progressive difficulty - Dense → Medium → Sparse rewards
4. Shared observation space - Same 183-dim observation for all games
5. Context vectors - Agent learns to adapt behavior per game
"""
import torch
import torch.optim as optim
import numpy as np
import random
from datetime import datetime
from collections import deque

from context_aware_agent import ContextAwareDQN, add_context_to_observation
from core.expanded_temporal_observer import ExpandedTemporalObserver
from core.enhanced_snake_game import EnhancedSnakeGame
from core.simple_pacman_game import SimplePacManGame
from core.simple_dungeon_game import SimpleDungeonGame
from core.local_view_game import LocalViewGame
from train_context_aware_advanced import PrioritizedReplayBuffer


class ExplorationTracker:
    """
    Track visited cells and provide exploration bonus
    """
    def __init__(self, bonus_value=0.5):
        self.visited_cells = set()
        self.bonus_value = bonus_value

    def visit(self, pos):
        """Mark cell as visited, return bonus if new"""
        if pos not in self.visited_cells:
            self.visited_cells.add(pos)
            return self.bonus_value
        return 0.0

    def reset(self):
        """Reset for new episode"""
        self.visited_cells.clear()

    def coverage(self, grid_size):
        """Calculate exploration coverage %"""
        total_cells = grid_size * grid_size
        return (len(self.visited_cells) / total_cells) * 100


class MultiGameRewards:
    """
    Unified reward system across all games with exploration bonus
    """
    def __init__(self, exploration_bonus=0.5):
        self.combo_count = 0
        self.steps_alive = 0
        self.exploration_tracker = ExplorationTracker(bonus_value=exploration_bonus)

    def reset(self):
        self.combo_count = 0
        self.steps_alive = 0
        self.exploration_tracker.reset()

    def calculate(self, base_reward, agent_pos, died=False):
        """
        Calculate enhanced reward with exploration bonus
        """
        total = base_reward

        # Exploration bonus
        exploration_reward = self.exploration_tracker.visit(agent_pos)
        total += exploration_reward

        # Survival bonus
        self.steps_alive += 1
        total += 0.1

        # Combo tracking for collection
        if base_reward >= 10:  # Collected reward (food/pellet/coin/treasure)
            self.combo_count += 1

        # Death penalty
        if died:
            total -= 50.0

        return total


def sample_game(episode, total_episodes=1000):
    """
    Sample game based on curriculum phase
    Returns: (game, game_name, context_vector, difficulty_level)
    """
    # Determine phase
    if episode < 300:
        # Phase 1: Dense rewards (Snake + Pac-Man)
        phase = 1
        weights = {'snake': 0.7, 'pacman': 0.3, 'local_view': 0.0, 'dungeon': 0.0}
    elif episode < 650:
        # Phase 2: Add medium rewards (Local View)
        phase = 2
        weights = {'snake': 0.4, 'pacman': 0.3, 'local_view': 0.3, 'dungeon': 0.0}
    else:
        # Phase 3: Add sparse rewards (Dungeon)
        phase = 3
        weights = {'snake': 0.3, 'pacman': 0.25, 'local_view': 0.25, 'dungeon': 0.2}

    # Sample game
    game_types = list(weights.keys())
    game_weights = list(weights.values())
    game_type = random.choices(game_types, weights=game_weights)[0]

    # Progressive difficulty within each game
    if episode < 200:
        difficulty = 0  # Easy
    elif episode < 500:
        difficulty = 1  # Medium
    else:
        difficulty = 2  # Hard

    # Create game instance
    if game_type == 'snake':
        if difficulty == 0:
            game = EnhancedSnakeGame(size=10, initial_pellets=3, max_pellets=7,
                                    obstacle_level=0, max_steps=200, max_total_food=15)
        elif difficulty == 1:
            game = EnhancedSnakeGame(size=15, initial_pellets=5, max_pellets=10,
                                    obstacle_level=1, max_steps=300, max_total_food=25)
        else:
            game = EnhancedSnakeGame(size=20, initial_pellets=7, max_pellets=12,
                                    obstacle_level=2, max_steps=400, max_total_food=40)
        context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)  # Snake context
        game_name = f"Snake-{difficulty}"

    elif game_type == 'pacman':
        # Start easy, gradually add ghosts
        ghost_level = min(difficulty, 2)  # 0-2 (no chasing yet)
        game = SimplePacManGame(size=20, num_pellets=30, ghost_level=ghost_level, max_steps=500)
        context = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)  # Balanced context
        game_name = f"PacMan-L{ghost_level}"

    elif game_type == 'local_view':
        # Start easy, gradually add enemies
        enemy_level = min(difficulty, 2)  # 0-2 enemies
        game = LocalViewGame(world_size=40, num_coins=20, enemy_level=enemy_level, max_steps=1000)
        context = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)  # Balanced context
        game_name = f"LocalView-L{enemy_level}"

    elif game_type == 'dungeon':
        # Start with no enemies, gradually add
        enemy_level = min(difficulty, 1)  # 0-1 enemies (sparse rewards are hard enough!)
        game = SimpleDungeonGame(size=20, num_treasures=3, enemy_level=enemy_level,
                                max_steps=600, history_length=2)
        context = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)  # Defensive context
        game_name = f"Dungeon-L{enemy_level}"

    return game, game_name, context, phase


def train_multi_game(episodes=1000, batch_size=64, learning_rate=1e-4):
    """Train foundation agent on multiple games"""

    print("=" * 80)
    print("MULTI-GAME CURRICULUM TRAINING - Foundation Agent")
    print("=" * 80)
    print(f"Total Episodes: {episodes}")
    print()
    print("CURRICULUM:")
    print("  Phase 1 (0-300):     Snake 70% + Pac-Man 30%")
    print("                       Learn: Dense rewards, moving obstacles")
    print()
    print("  Phase 2 (300-650):   Snake 40% + Pac-Man 30% + Local View 30%")
    print("                       Learn: Large worlds, medium density")
    print()
    print("  Phase 3 (650-1000):  Snake 30% + Pac-Man 25% + Local View 25% + Dungeon 20%")
    print("                       Learn: Sparse rewards, maze navigation")
    print()
    print("KEY FEATURES:")
    print("  1. Exploration bonus (+0.5 per new cell)")
    print("  2. Shared observation space (183-dim rays)")
    print("  3. Context-aware adaptation")
    print("  4. Progressive difficulty within each game")
    print()

    # Initialize
    device = torch.device('cpu')
    policy_net = ContextAwareDQN(obs_dim=183, action_dim=4).to(device)
    target_net = ContextAwareDQN(obs_dim=183, action_dim=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = PrioritizedReplayBuffer(capacity=100000)  # Larger for multi-game

    # Observer (shared across all games!)
    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)

    # Reward system with exploration bonus
    reward_system = MultiGameRewards(exploration_bonus=0.5)

    # Training parameters
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = episodes * 0.7

    # Tracking
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    game_type_counts = {'snake': 0, 'pacman': 0, 'local_view': 0, 'dungeon': 0}
    game_type_scores = {'snake': [], 'pacman': [], 'local_view': [], 'dungeon': []}

    for episode in range(1, episodes + 1):
        # Sample game from curriculum
        game, game_name, context, phase = sample_game(episode, episodes)

        # Track game type (handle LocalView -> local_view conversion)
        game_type_raw = game_name.split('-')[0].lower()
        # Map to dictionary keys
        game_type_map = {'snake': 'snake', 'pacman': 'pacman',
                        'localview': 'local_view', 'dungeon': 'dungeon'}
        game_type = game_type_map.get(game_type_raw, game_type_raw)
        game_type_counts[game_type] += 1

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
            # Observe (same observer for all games!)
            obs = observer.observe(state)
            obs_with_context = add_context_to_observation(obs, context)

            # Select action
            action = policy_net.get_action(obs_with_context, epsilon=epsilon)

            # Step
            prev_lives = getattr(game, 'lives', 1)
            next_state, base_reward, done = game.step(action)

            # Check death
            current_lives = getattr(game, 'lives', 1)
            died = (prev_lives > current_lives)

            # Get agent position (different for each game)
            if hasattr(game, 'snake') and game.snake:
                agent_pos = game.snake[0]
            elif hasattr(game, 'pacman_pos'):
                agent_pos = game.pacman_pos
            elif hasattr(game, 'player_pos'):
                agent_pos = game.player_pos
            elif hasattr(game, 'agent_pos'):
                agent_pos = game.agent_pos
            else:
                agent_pos = (0, 0)

            # Enhanced reward with exploration bonus
            enhanced_reward = reward_system.calculate(base_reward, agent_pos, died)

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

                # Update priorities
                priorities = td_errors.detach().cpu().numpy()
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
        game_type_scores[game_type].append(game.score)

        # Update target network
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Log progress
        if episode % 50 == 0 or episode == episodes:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_score = np.mean(episode_scores[-100:])
            avg_length = np.mean(episode_lengths[-100:])

            print(f"\nEpisode {episode}/{episodes} [Phase {phase}]")
            print(f"  Game: {game_name}")
            print(f"  Avg Reward (100): {avg_reward:.2f}")
            print(f"  Avg Score (100): {avg_score:.2f}")
            print(f"  Avg Length (100): {avg_length:.1f}")
            print(f"  Epsilon: {epsilon:.3f}")
            print(f"  Buffer: {len(replay_buffer)}")

            # Game distribution
            print(f"  Game Distribution:")
            total = sum(game_type_counts.values())
            for gtype, count in game_type_counts.items():
                pct = (count / total * 100) if total > 0 else 0
                avg_score_type = np.mean(game_type_scores[gtype][-20:]) if game_type_scores[gtype] else 0
                print(f"    {gtype:12} {count:4} ({pct:4.1f}%)  Avg Score: {avg_score_type:.2f}")

            # Exploration coverage
            grid_size = getattr(game, 'size', getattr(game, 'world_size', 20))
            coverage = reward_system.exploration_tracker.coverage(grid_size)
            print(f"  Exploration: {coverage:.1f}% of map")

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    checkpoints_dir = os.path.join(repo_root, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    save_path = os.path.join(checkpoints_dir, f"multi_game_{timestamp}_policy.pth")

    torch.save({
        'episode': episodes,
        'policy_net': policy_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_scores': episode_scores,
        'episode_lengths': episode_lengths,
        'game_type_counts': game_type_counts,
        'game_type_scores': game_type_scores,
        'curriculum': {
            'phase_1': '0-300: Snake 70% + PacMan 30%',
            'phase_2': '300-650: Snake 40% + PacMan 30% + LocalView 30%',
            'phase_3': '650-1000: Snake 30% + PacMan 25% + LocalView 25% + Dungeon 20%'
        },
        'features': {
            'exploration_bonus': 0.5,
            'multi_game': True,
            'curriculum': True,
            'context_aware': True
        }
    }, save_path)

    print(f"\n[OK] Model saved: {save_path}")
    print()
    print("=" * 80)
    print("TRAINING COMPLETE - Multi-Game Foundation Agent")
    print("=" * 80)
    print(f"Final Avg Score (100): {np.mean(episode_scores[-100:]):.2f}")
    print(f"Final Avg Reward (100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final Avg Length (100): {np.mean(episode_lengths[-100:]):.1f}")
    print()
    print("Game-Specific Performance (last 100 episodes):")
    for gtype in ['snake', 'pacman', 'local_view', 'dungeon']:
        recent_scores = game_type_scores[gtype][-100:]
        if recent_scores:
            avg = np.mean(recent_scores)
            max_score = max(recent_scores)
            print(f"  {gtype:12} Avg: {avg:5.2f}  Max: {max_score:3.0f}  Episodes: {len(recent_scores)}")
    print()
    print("EXPECTED IMPROVEMENTS vs Single-Game Training:")
    print("  - Pac-Man win rate: 40% → 60-70%")
    print("  - Dungeon win rate: 0% → 30-50%")
    print("  - Local View win rate: 0% → 40-60%")
    print("  - Better exploration (systematic vs random)")
    print("  - Handles sparse rewards")
    print("  - Adapts to different world sizes")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train_multi_game(
        episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

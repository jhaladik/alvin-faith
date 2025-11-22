"""
Enhanced Multi-Game Training - Incorporating train_snake_improved.py Success Factors

Takes the best practices from train_snake_improved.py and applies them to multi-game curriculum:
1. NET progress tracking (anti-circling)
2. Proximity bonus (encourages approach)
3. Stagnation penalty (prevents wandering)
4. Stronger collection rewards (100 + combo*20)
5. Prioritized replay for collisions (2x priority)
6. Exploration bonus (for sparse rewards)

CURRICULUM (same as before):
- Phase 1 (0-300):     Snake 70% + Pac-Man 30%
- Phase 2 (300-650):   Snake 40% + Pac-Man 30% + Local View 30%
- Phase 3 (650-1000):  Snake 30% + Pac-Man 25% + Local View 25% + Dungeon 20%
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


class EnhancedMultiGameRewards:
    """
    Enhanced reward system combining exploration bonus with train_snake_improved.py techniques
    """
    def __init__(self, exploration_bonus=0.5):
        self.combo_count = 0
        self.steps_alive = 0
        self.steps_since_collection = 0
        self.reward_dist_history = deque(maxlen=5)  # Track progress over 5 steps
        self.visited_cells = set()
        self.exploration_bonus = exploration_bonus

    def reset(self):
        self.combo_count = 0
        self.steps_alive = 0
        self.steps_since_collection = 0
        self.reward_dist_history.clear()
        self.visited_cells.clear()

    def calculate(self, base_reward, agent_pos, nearest_reward_dist=None,
                  min_wall_dist=None, died=False):
        """
        Enhanced reward calculation with all success factors
        """
        total = base_reward

        # 1. COLLECTION REWARD - Much stronger (100 + combo*20)
        if base_reward >= 10:  # Collected something (food/pellet/coin/treasure)
            collection_bonus = 100.0 + (self.combo_count * 20.0)
            total = collection_bonus  # Replace base reward
            self.combo_count += 1
            self.steps_since_collection = 0

        # 2. EXPLORATION BONUS - Rewards visiting new cells
        if agent_pos not in self.visited_cells:
            self.visited_cells.add(agent_pos)
            total += self.exploration_bonus

        # 3. SURVIVAL BONUS
        self.steps_alive += 1
        self.steps_since_collection += 1
        total += 0.1

        # 4. NET PROGRESS TRACKING (Anti-circling)
        if nearest_reward_dist is not None:
            self.reward_dist_history.append(nearest_reward_dist)

            if len(self.reward_dist_history) >= 5:
                # Reward actual progress over 5 steps
                initial_dist = self.reward_dist_history[0]
                current_dist = nearest_reward_dist
                net_progress = initial_dist - current_dist

                if net_progress > 0:
                    total += net_progress * 2.0  # Reward real progress
                elif net_progress < -2:
                    total -= 1.0  # Penalty for moving away

            # 5. PROXIMITY BONUS - Big reward when adjacent
            if nearest_reward_dist == 1:
                total += 5.0

        # 6. DANGER ZONE SHAPING (Weak penalty near walls)
        if min_wall_dist is not None and min_wall_dist < 1.2:
            danger_penalty = -0.5 * (1.0 - min_wall_dist / 1.2)
            total += danger_penalty

        # 7. STAGNATION PENALTY - Prevents wandering
        if self.steps_since_collection > 30:
            total -= 0.5

        # 8. DEATH PENALTY
        if died:
            total -= 100.0

        return total

    def coverage(self, grid_size):
        """Calculate exploration coverage %"""
        total_cells = grid_size * grid_size
        return (len(self.visited_cells) / total_cells) * 100


def get_nearest_reward_distance(agent_pos, reward_positions):
    """Calculate Manhattan distance to nearest reward"""
    if not reward_positions:
        return None

    distances = [abs(agent_pos[0] - r[0]) + abs(agent_pos[1] - r[1])
                 for r in reward_positions]
    return min(distances)


def get_min_wall_distance(obs):
    """Extract minimum wall distance from observation"""
    # Extract wall distances from observation (every 3rd element starting at index 2)
    wall_dists = obs[2::3][:16]  # 16 rays
    min_wall_dist_normalized = wall_dists.min()
    min_wall_dist_tiles = min_wall_dist_normalized * 15  # Denormalize (ray_length=15)
    return min_wall_dist_tiles


def sample_game(episode, total_episodes=1000):
    """Sample game based on curriculum phase"""
    # Determine phase
    if episode < 300:
        phase = 1
        weights = {'snake': 0.7, 'pacman': 0.3, 'local_view': 0.0, 'dungeon': 0.0}
    elif episode < 650:
        phase = 2
        weights = {'snake': 0.4, 'pacman': 0.3, 'local_view': 0.3, 'dungeon': 0.0}
    else:
        phase = 3
        weights = {'snake': 0.3, 'pacman': 0.25, 'local_view': 0.25, 'dungeon': 0.2}

    # Sample game
    game_types = list(weights.keys())
    game_weights = list(weights.values())
    game_type = random.choices(game_types, weights=game_weights)[0]

    # Progressive difficulty
    if episode < 200:
        difficulty = 0
    elif episode < 500:
        difficulty = 1
    else:
        difficulty = 2

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
        context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        game_name = f"Snake-{difficulty}"

    elif game_type == 'pacman':
        ghost_level = min(difficulty, 2)
        game = SimplePacManGame(size=20, num_pellets=30, ghost_level=ghost_level, max_steps=500)
        context = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        game_name = f"PacMan-L{ghost_level}"

    elif game_type == 'local_view':
        enemy_level = min(difficulty, 2)
        game = LocalViewGame(world_size=40, num_coins=20, enemy_level=enemy_level, max_steps=1000)
        context = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        game_name = f"LocalView-L{enemy_level}"

    elif game_type == 'dungeon':
        enemy_level = min(difficulty, 1)
        game = SimpleDungeonGame(size=20, num_treasures=3, enemy_level=enemy_level,
                                max_steps=600, history_length=2)
        context = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        game_name = f"Dungeon-L{enemy_level}"

    return game, game_name, context, phase


def train_multi_game_enhanced(episodes=1000, batch_size=64, learning_rate=1e-4):
    """Train foundation agent with enhanced rewards"""

    print("=" * 80)
    print("ENHANCED MULTI-GAME TRAINING - Best Practices Edition")
    print("=" * 80)
    print(f"Total Episodes: {episodes}")
    print()
    print("ENHANCEMENTS from train_snake_improved.py:")
    print("  1. NET progress tracking (anti-circling)")
    print("  2. Proximity bonus (+5 when adjacent)")
    print("  3. Stagnation penalty (-0.5 after 30 steps)")
    print("  4. Stronger collection (100 + combo*20)")
    print("  5. Prioritized replay for collisions (2x)")
    print("  6. Exploration bonus (+0.5 per new cell)")
    print("  7. Danger zone shaping (weak penalty)")
    print()

    # Initialize
    device = torch.device('cpu')
    policy_net = ContextAwareDQN(obs_dim=183, action_dim=4).to(device)
    target_net = ContextAwareDQN(obs_dim=183, action_dim=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = PrioritizedReplayBuffer(capacity=100000)

    # Observer
    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)

    # Enhanced reward system
    reward_system = EnhancedMultiGameRewards(exploration_bonus=0.5)

    # Training parameters
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = episodes * 0.7  # Slower decay like train_snake_improved

    # Tracking
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    game_type_counts = {'snake': 0, 'pacman': 0, 'local_view': 0, 'dungeon': 0}
    game_type_scores = {'snake': [], 'pacman': [], 'local_view': [], 'dungeon': []}

    for episode in range(1, episodes + 1):
        # Sample game from curriculum
        game, game_name, context, phase = sample_game(episode, episodes)

        # Track game type
        game_type_raw = game_name.split('-')[0].lower()
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
            # Observe
            obs = observer.observe(state)
            obs_with_context = add_context_to_observation(obs, context)

            # Get agent position (game-specific)
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

            # Get reward positions (game-specific)
            if hasattr(game, 'food_positions'):
                reward_positions = list(game.food_positions)
            elif hasattr(game, 'pellets'):
                reward_positions = list(game.pellets)
            elif hasattr(game, 'coins'):
                reward_positions = list(game.coins)
            elif hasattr(game, 'treasures'):
                reward_positions = list(game.treasures)
            else:
                reward_positions = []

            # Calculate nearest reward distance
            nearest_reward_dist = get_nearest_reward_distance(agent_pos, reward_positions)

            # Calculate min wall distance
            min_wall_dist = get_min_wall_distance(obs)

            # Select action
            action = policy_net.get_action(obs_with_context, epsilon=epsilon)

            # Step
            prev_lives = getattr(game, 'lives', 1)
            next_state, base_reward, done = game.step(action)

            # Check death
            current_lives = getattr(game, 'lives', 1)
            died = (prev_lives > current_lives)

            # Enhanced reward with ALL success factors
            enhanced_reward = reward_system.calculate(
                base_reward, agent_pos, nearest_reward_dist,
                min_wall_dist, died
            )

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

                # PRIORITIZED REPLAY: 2x priority for collisions (like train_snake_improved)
                priorities = td_errors.detach().cpu().numpy()
                for i, reward in enumerate(rewards):
                    if reward < -50:  # Collision experience
                        priorities[i] *= 2.0  # Double the priority!

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
            coverage = reward_system.coverage(grid_size)
            print(f"  Exploration: {coverage:.1f}% of map")

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    checkpoints_dir = os.path.join(repo_root, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    save_path = os.path.join(checkpoints_dir, f"multi_game_enhanced_{timestamp}_policy.pth")

    torch.save({
        'episode': episodes,
        'policy_net': policy_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_scores': episode_scores,
        'episode_lengths': episode_lengths,
        'game_type_counts': game_type_counts,
        'game_type_scores': game_type_scores,
        'enhancements': {
            'net_progress_tracking': True,
            'proximity_bonus': True,
            'stagnation_penalty': True,
            'stronger_collection': True,
            'prioritized_collisions': True,
            'exploration_bonus': True,
            'danger_zone_shaping': True
        }
    }, save_path)

    print(f"\n[OK] Model saved: {save_path}")
    print()
    print("=" * 80)
    print("TRAINING COMPLETE - Enhanced Multi-Game Foundation Agent")
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
    print("EXPECTED IMPROVEMENTS vs Basic Multi-Game:")
    print("  - FASTER learning (stronger rewards, prioritized replay)")
    print("  - HIGHER scores (net progress tracking prevents circling)")
    print("  - BETTER exploration (combo of exploration bonus + stagnation penalty)")
    print("  - Pac-Man: 80-90% collection (vs 70%)")
    print("  - Dungeon: 50-70% collection (vs 30-40%)")
    print("  - Local View: 70-90% collection (vs 50-60%)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train_multi_game_enhanced(
        episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

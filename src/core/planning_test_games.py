"""
Test Planning Agent on Snake, Pac-Man, and Dungeon games.

Tests the COMPLETE system:
1. Temporal understanding (current + delta features)
2. Hierarchical priorities (4 Q-heads)
3. World model prediction (imagine futures)
4. Planning (simulate before acting)

Usage:
    python planning_test_games.py <model_base_path>
    python planning_test_games.py checkpoints/planning_agent_20251117_220042_best
"""
import torch
import numpy as np
import random
import argparse
from .temporal_observer import TemporalFlowObserver
from .temporal_agent import TemporalHierarchicalDQN
from .world_model import WorldModelNetwork, PlanningAgent


class PlanningGameAgent:
    """Agent using temporal understanding + world model planning."""

    def __init__(self, policy_path=None, world_model_path=None, planning_horizon=5):
        self.policy_net = TemporalHierarchicalDQN(obs_dim=92, action_dim=4)
        self.world_model = WorldModelNetwork(state_dim=92, action_dim=4)

        if policy_path:
            checkpoint = torch.load(policy_path, map_location='cpu')
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            print(f"Loaded policy: {policy_path}")

        if world_model_path:
            checkpoint = torch.load(world_model_path, map_location='cpu')
            self.world_model.load_state_dict(checkpoint['model'])
            print(f"Loaded world model: {world_model_path}")

        self.planner = PlanningAgent(
            self.policy_net,
            self.world_model,
            planning_horizon=planning_horizon
        )

        self.use_planning = True
        self.planning_frequency = 0.5
        self.observer = TemporalFlowObserver()

        self.planning_count = 0
        self.reactive_count = 0

    def get_action(self, game_state):
        obs = self.observer.observe(game_state)

        if self.use_planning and random.random() < self.planning_frequency:
            action, value = self.planner.plan_action(obs, num_simulations=12)
            self.planning_count += 1
        else:
            action = self.policy_net.get_action(obs, epsilon=0.0)
            self.reactive_count += 1

        return action

    def reset_observer(self):
        self.observer.reset()


# =============================================================================
# GAME ENVIRONMENTS
# =============================================================================

class SnakeGame:
    def __init__(self, size=20, num_pellets=10):  # MATCH TRAINING: 20x20 grid, multiple pellets
        self.size = size
        self.num_pellets = num_pellets
        self.reset()

    def reset(self):
        center = self.size // 2
        self.snake = [(center, center)]
        self.food_positions = self._place_all_food()  # MATCH TRAINING: Multiple pellets
        self.direction = (0, -1)
        self.score = 0
        self.steps = 0
        self.max_steps = 1000  # MATCH TRAINING: 1000 steps
        self.lives = 3  # MATCH TRAINING: 3 lives
        self.total_collected = 0  # Track total food collected
        self.done = False
        return self._get_game_state()

    def _place_all_food(self):
        """Place multiple food pellets at once (MATCH TRAINING)"""
        food = set()
        attempts = 0
        while len(food) < self.num_pellets and attempts < 5000:
            pos = (random.randint(1, self.size-2), random.randint(1, self.size-2))
            if pos not in self.snake and pos not in food:
                food.add(pos)
            attempts += 1
        return food

    def _get_game_state(self):
        walls = set()
        for i in range(self.size):
            walls.add((i, 0))
            walls.add((i, self.size-1))
            walls.add((0, i))
            walls.add((self.size-1, i))

        # FIX: Snake body segments should NOT be treated as external entities!
        # The agent's own body is not a threat to avoid - it's part of self-collision
        entities = []
        # Don't add snake body as entities - this was causing wrong context detection

        return {
            'agent_pos': self.snake[0],
            'walls': walls,
            'rewards': list(self.food_positions),  # MATCH TRAINING: Multiple pellets
            'entities': entities,
            'grid_size': (self.size, self.size),
            'score': self.score,
            'done': self.done
        }

    def step(self, action):
        if self.done:
            return self._get_game_state(), 0, True

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_dir = directions[action]
        if (new_dir[0] + self.direction[0] != 0) or (new_dir[1] + self.direction[1] != 0):
            self.direction = new_dir

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # MATCH TRAINING: Reward movement (+0.1)
        reward = 0.1

        # Wall collision
        if (new_head[0] <= 0 or new_head[0] >= self.size-1 or
            new_head[1] <= 0 or new_head[1] >= self.size-1):
            # MATCH TRAINING: Lives system instead of instant death
            self.lives -= 1
            reward = -50.0  # MATCH TRAINING: -50 penalty
            if self.lives <= 0:
                self.done = True
                reward -= 100.0  # MATCH TRAINING: -100 for game over
                return self._get_game_state(), reward, True
            else:
                # Respawn at center
                center = self.size // 2
                self.snake = [(center, center)]
                self.direction = (0, -1)
                return self._get_game_state(), reward, False

        # Self collision
        if new_head in self.snake:
            self.lives -= 1
            reward = -50.0
            if self.lives <= 0:
                self.done = True
                reward -= 100.0
                return self._get_game_state(), reward, True
            else:
                center = self.size // 2
                self.snake = [(center, center)]
                self.direction = (0, -1)
                return self._get_game_state(), reward, False

        self.snake.insert(0, new_head)

        # Food collection - MATCH TRAINING: Check set of pellets, don't respawn
        if new_head in self.food_positions:
            self.food_positions.remove(new_head)
            self.score += 1
            self.total_collected += 1
            reward = 20.0  # MATCH TRAINING: +20 for food
        else:
            self.snake.pop()

        # Victory condition - MATCH TRAINING: All pellets collected
        if len(self.food_positions) == 0:
            self.done = True
            reward += 200.0  # MATCH TRAINING: +200 victory bonus
            return self._get_game_state(), reward, True

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_game_state(), reward, self.done


class PacManGame:
    def __init__(self, size=20):  # MATCH TRAINING: 20x20 grid
        self.size = size
        self.reset()

    def reset(self):
        center = self.size // 2
        self.pacman_pos = (center, center)

        self.walls = set()
        for i in range(self.size):
            self.walls.add((i, 0))
            self.walls.add((i, self.size-1))
            self.walls.add((0, i))
            self.walls.add((self.size-1, i))

        for _ in range(int(self.size * self.size * 0.15)):
            x = random.randint(1, self.size-2)
            y = random.randint(1, self.size-2)
            if (x, y) != self.pacman_pos:
                self.walls.add((x, y))

        self.pellets = set()
        for x in range(1, self.size-1):
            for y in range(1, self.size-1):
                if (x, y) not in self.walls and (x, y) != self.pacman_pos:
                    if random.random() < 0.3:
                        self.pellets.add((x, y))

        self.ghosts = []
        for _ in range(3):
            while True:
                gx = random.randint(1, self.size-2)
                gy = random.randint(1, self.size-2)
                if (gx, gy) not in self.walls and abs(gx - center) > 3:
                    self.ghosts.append({'pos': (gx, gy), 'dir': (0, 0)})
                    break

        self.score = 0
        self.steps = 0
        self.max_steps = 1000  # MATCH TRAINING: 1000 steps
        self.lives = 3  # MATCH TRAINING: 3 lives
        self.total_collected = 0  # Track total pellets
        self.done = False

        return self._get_game_state()

    def _get_game_state(self):
        entities = []
        for ghost in self.ghosts:
            entities.append({
                'pos': ghost['pos'],
                'velocity': ghost['dir'],
                'danger': 1.0
            })

        return {
            'agent_pos': self.pacman_pos,
            'walls': self.walls,
            'rewards': list(self.pellets),
            'entities': entities,
            'grid_size': (self.size, self.size),  # FIX: Add grid_size for observer
            'score': self.score,
            'done': self.done
        }

    def step(self, action):
        if self.done:
            return self._get_game_state(), 0, True

        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_pos = (self.pacman_pos[0] + dx, self.pacman_pos[1] + dy)

        # MATCH TRAINING: Reward movement (+0.1)
        reward = 0.1

        if new_pos not in self.walls:
            self.pacman_pos = new_pos

        # Pellet collection
        if self.pacman_pos in self.pellets:
            self.pellets.remove(self.pacman_pos)
            self.score += 1
            self.total_collected += 1
            reward = 20.0  # MATCH TRAINING: +20 for pellets

        # Move ghosts
        for ghost in self.ghosts:
            gx, gy = ghost['pos']
            px, py = self.pacman_pos

            possible = []
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                new_gpos = (gx + ddx, gy + ddy)
                if new_gpos not in self.walls:
                    dist = abs(new_gpos[0] - px) + abs(new_gpos[1] - py)
                    possible.append((dist, new_gpos, (ddx, ddy)))

            if possible:
                possible.sort()
                if random.random() < 0.7:
                    ghost['pos'] = possible[0][1]
                    ghost['dir'] = possible[0][2]
                else:
                    choice = random.choice(possible)
                    ghost['pos'] = choice[1]
                    ghost['dir'] = choice[2]

        # Ghost collision - MATCH TRAINING: Lives system
        for ghost in self.ghosts:
            if ghost['pos'] == self.pacman_pos:
                self.lives -= 1
                reward = -50.0  # MATCH TRAINING: -50 penalty
                if self.lives <= 0:
                    self.done = True
                    reward -= 100.0  # MATCH TRAINING: -100 for game over
                    return self._get_game_state(), reward, True
                else:
                    # Respawn at center
                    center = self.size // 2
                    self.pacman_pos = (center, center)
                    return self._get_game_state(), reward, False

        # Victory condition - MATCH TRAINING: +200 bonus
        if len(self.pellets) == 0:
            self.done = True
            reward = 200.0  # MATCH TRAINING: +200 victory bonus

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_game_state(), reward, self.done


class DungeonGame:
    def __init__(self, size=20):  # MATCH TRAINING: 20x20 grid
        self.size = size
        self.reset()

    def reset(self):
        self.player_pos = (1, 1)

        self.walls = set()
        for i in range(self.size):
            self.walls.add((i, 0))
            self.walls.add((i, self.size-1))
            self.walls.add((0, i))
            self.walls.add((self.size-1, i))

        for _ in range(int(self.size * self.size * 0.25)):
            x = random.randint(1, self.size-2)
            y = random.randint(1, self.size-2)
            if (x, y) != self.player_pos:
                self.walls.add((x, y))

        while True:
            tx = random.randint(self.size//2, self.size-2)
            ty = random.randint(self.size//2, self.size-2)
            if (tx, ty) not in self.walls:
                self.treasure = (tx, ty)
                break

        self.enemies = []
        for _ in range(2):
            while True:
                ex = random.randint(3, self.size-3)
                ey = random.randint(3, self.size-3)
                if (ex, ey) not in self.walls:
                    self.enemies.append({'pos': (ex, ey), 'dir': (0, 0)})
                    break

        self.score = 0
        self.steps = 0
        self.max_steps = 1000  # MATCH TRAINING: 1000 steps
        self.lives = 3  # MATCH TRAINING: 3 lives
        self.total_collected = 0  # Track total treasures collected
        self.done = False

        return self._get_game_state()

    def _get_game_state(self):
        entities = []
        for enemy in self.enemies:
            entities.append({
                'pos': enemy['pos'],
                'velocity': enemy['dir'],
                'danger': 1.0
            })

        return {
            'agent_pos': self.player_pos,
            'walls': self.walls,
            'rewards': [self.treasure],
            'entities': entities,
            'grid_size': (self.size, self.size),  # FIX: Add grid_size for observer
            'score': self.score,
            'done': self.done
        }

    def step(self, action):
        if self.done:
            return self._get_game_state(), 0, True

        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

        # MATCH TRAINING: Reward movement (+0.1)
        reward = 0.1

        if new_pos not in self.walls:
            self.player_pos = new_pos

        # Treasure collection - MATCH TRAINING: +200 for treasure
        if self.player_pos == self.treasure:
            self.score += 10
            self.total_collected += 1
            reward = 200.0  # MATCH TRAINING: +200 victory bonus
            self.done = True
            return self._get_game_state(), reward, True

        # Move enemies
        for enemy in self.enemies:
            ex, ey = enemy['pos']

            possible = []
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                new_epos = (ex + ddx, ey + ddy)
                if new_epos not in self.walls:
                    possible.append((new_epos, (ddx, ddy)))

            if possible:
                choice = random.choice(possible)
                enemy['pos'] = choice[0]
                enemy['dir'] = choice[1]

        # Enemy collision - MATCH TRAINING: Lives system
        for enemy in self.enemies:
            if enemy['pos'] == self.player_pos:
                self.lives -= 1
                reward = -50.0  # MATCH TRAINING: -50 penalty
                if self.lives <= 0:
                    self.done = True
                    reward -= 100.0  # MATCH TRAINING: -100 for game over
                    return self._get_game_state(), reward, True
                else:
                    # Respawn at start position
                    self.player_pos = (1, 1)
                    return self._get_game_state(), reward, False

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_game_state(), reward, self.done


def test_all_games(agent, num_episodes=10):
    print("=" * 60)
    print("TESTING PLANNING AGENT")
    print("=" * 60)
    print()

    games = {
        'Snake': SnakeGame(size=20),  # MATCH TRAINING: 20x20
        'Pac-Man': PacManGame(size=20),  # MATCH TRAINING: 20x20
        'Dungeon': DungeonGame(size=20)  # MATCH TRAINING: 20x20
    }

    results = {}

    for game_name, game in games.items():
        print(f"Testing {game_name}...")
        scores = []

        for episode in range(num_episodes):
            game.reset()
            agent.reset_observer()

            game_state = game._get_game_state()
            total_reward = 0

            while not game_state['done']:
                action = agent.get_action(game_state)
                game_state, reward, done = game.step(action)
                total_reward += reward

            scores.append(game_state['score'])

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        results[game_name] = (avg_score, std_score)

        print(f"  {game_name}: {avg_score:.2f} +/- {std_score:.2f}")
        print(f"  Individual: {scores}")
        print()

    return results


def main():
    parser = argparse.ArgumentParser(description='Test Planning Agent on Games')
    parser.add_argument('model_path', nargs='?', default=None, help='Base path to model (without _policy.pth)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of test episodes per game')
    parser.add_argument('--planning-freq', type=float, default=0.5, help='Planning frequency (0-1)')
    parser.add_argument('--horizon', type=int, default=5, help='Planning horizon')

    args = parser.parse_args()

    print("=" * 60)
    print("PLANNING AGENT GAME TESTER")
    print("=" * 60)
    print()

    policy_path = None
    world_model_path = None

    if args.model_path:
        policy_path = f"{args.model_path}_policy.pth"
        world_model_path = f"{args.model_path}_world_model.pth"
        print(f"Loading models from: {args.model_path}")
    else:
        print("No model path provided - using untrained agent")
        print("Usage: python planning_test_games.py <model_base_path>")
        print()

    agent = PlanningGameAgent(
        policy_path=policy_path,
        world_model_path=world_model_path,
        planning_horizon=args.horizon
    )

    agent.planning_frequency = args.planning_freq
    print(f"Planning frequency: {args.planning_freq}")
    print(f"Planning horizon: {args.horizon}")
    print()

    print(f"\n=== TEST: {int(args.planning_freq*100)}% Planning ===")
    results = test_all_games(agent, num_episodes=args.episodes)

    print("Planning/Reactive decisions:", agent.planning_count, "/", agent.reactive_count)
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for game, (avg, std) in results.items():
        print(f"{game}: {avg:.2f} +/- {std:.2f}")


if __name__ == '__main__':
    main()

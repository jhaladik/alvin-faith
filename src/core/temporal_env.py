"""
Training Environment using Temporal Flow Observer
Random mazes with temporal understanding for foundation agent.
"""
import numpy as np
import random
from .temporal_observer import TemporalFlowObserver


class TemporalRandom2DEnv:
    """
    Random 2D environment with temporal flow observations.
    Each episode generates NEW random maze for transfer learning.
    """

    def __init__(self, grid_size=(20, 20), num_entities=4, num_rewards=10,
                 maze_complexity=0.3, entity_speed=0.7):
        """
        Args:
            grid_size: (width, height)
            num_entities: Dangerous entities (0-6 for variety)
            num_rewards: Sparse rewards (10 instead of 50 for target-seeking)
            maze_complexity: Wall density
            entity_speed: How often entities move
        """
        self.grid_width, self.grid_height = grid_size
        self.num_entities = num_entities
        self.num_rewards = num_rewards
        self.maze_complexity = maze_complexity
        self.entity_speed = entity_speed

        # State
        self.agent_pos = None
        self.walls = set()
        self.entities = []
        self.rewards = set()
        self.last_action = -1

        # Episode tracking
        self.steps = 0
        self.max_steps = 1000
        self.total_collected = 0
        self.lives = 3

        # Temporal flow observer (key difference!)
        self.observer = TemporalFlowObserver(num_rays=8, ray_length=10)
        self.obs_dim = self.observer.obs_dim

    def reset(self):
        """Reset with NEW random maze"""
        self.steps = 0
        self.total_collected = 0
        self.lives = 3
        self.last_action = -1

        # Generate new maze
        self._generate_maze()

        # Place agent
        self.agent_pos = self._find_open_position()

        # Place sparse rewards (10, not 50!)
        self._place_rewards()

        # Spawn entities
        self._spawn_entities()

        # Reset observer (clears temporal history)
        self.observer.reset()

        return self._get_observation()

    def _generate_maze(self):
        """Generate random maze"""
        self.walls = set()

        # Boundary walls
        for x in range(self.grid_width):
            self.walls.add((x, 0))
            self.walls.add((x, self.grid_height - 1))
        for y in range(self.grid_height):
            self.walls.add((0, y))
            self.walls.add((self.grid_width - 1, y))

        # Internal walls
        if self.maze_complexity > 0:
            num_structures = int(self.maze_complexity * 15)

            for _ in range(num_structures):
                structure = random.choice(['horizontal', 'vertical', 'box'])

                if structure == 'horizontal':
                    x = random.randint(2, self.grid_width - 4)
                    y = random.randint(2, self.grid_height - 3)
                    length = random.randint(3, 6)
                    for dx in range(length):
                        if x + dx < self.grid_width - 1:
                            self.walls.add((x + dx, y))

                elif structure == 'vertical':
                    x = random.randint(2, self.grid_width - 3)
                    y = random.randint(2, self.grid_height - 4)
                    length = random.randint(3, 6)
                    for dy in range(length):
                        if y + dy < self.grid_height - 1:
                            self.walls.add((x, y + dy))

                elif structure == 'box':
                    cx = random.randint(3, self.grid_width - 5)
                    cy = random.randint(3, self.grid_height - 5)
                    size = random.randint(2, 3)
                    for dx in range(size):
                        self.walls.add((cx + dx, cy))
                        self.walls.add((cx + dx, cy + size - 1))
                    for dy in range(size):
                        self.walls.add((cx, cy + dy))
                        self.walls.add((cx + size - 1, cy + dy))

    def _find_open_position(self):
        """Find random open position"""
        for _ in range(1000):
            x = random.randint(1, self.grid_width - 2)
            y = random.randint(1, self.grid_height - 2)
            if (x, y) not in self.walls:
                return (x, y)
        return (self.grid_width // 2, self.grid_height // 2)

    def _place_rewards(self):
        """Place sparse rewards"""
        self.rewards = set()
        attempts = 0
        while len(self.rewards) < self.num_rewards and attempts < 5000:
            x = random.randint(1, self.grid_width - 2)
            y = random.randint(1, self.grid_height - 2)
            if (x, y) not in self.walls and (x, y) != self.agent_pos:
                self.rewards.add((x, y))
            attempts += 1

    def _spawn_entities(self):
        """Spawn dangerous entities"""
        self.entities = []
        for _ in range(self.num_entities):
            pos = self._find_open_position()
            # Ensure not too close to agent
            while abs(pos[0] - self.agent_pos[0]) + abs(pos[1] - self.agent_pos[1]) < 5:
                pos = self._find_open_position()

            self.entities.append({
                'pos': pos,
                'velocity': (0, 0),
                'danger': 0.8,
                'behavior': random.choice(['chase', 'patrol', 'random'])
            })

    def step(self, action):
        """Take action, return temporal observation"""
        self.steps += 1
        reward = 0.0
        done = False
        info = {'died': False, 'collected_reward': False}

        # Move agent
        old_pos = self.agent_pos
        new_pos = self._move(self.agent_pos, action)

        if new_pos != old_pos:
            self.agent_pos = new_pos
            reward += 0.1  # Reward for moving
        else:
            reward -= 0.3  # Penalty for hitting wall

        self.last_action = action

        # Check reward collection
        if self.agent_pos in self.rewards:
            self.rewards.remove(self.agent_pos)
            reward += 20.0  # Larger reward for sparse targets
            self.total_collected += 1
            info['collected_reward'] = True

        # Check entity collision (before entities move)
        for entity in self.entities:
            if entity['pos'] == self.agent_pos and entity['danger'] > 0.5:
                reward -= 50.0
                self.lives -= 1
                info['died'] = True
                self.agent_pos = self._find_open_position()

        # Move entities
        self._update_entities()

        # Check collision again (after entities move)
        for entity in self.entities:
            if entity['pos'] == self.agent_pos and entity['danger'] > 0.5:
                reward -= 50.0
                self.lives -= 1
                info['died'] = True
                self.agent_pos = self._find_open_position()

        # Episode end conditions
        if self.lives <= 0:
            done = True
            reward -= 100.0

        if len(self.rewards) == 0:
            done = True
            reward += 200.0  # Victory!

        if self.steps >= self.max_steps:
            done = True

        # Proximity danger penalty (REDUCED - hierarchical rewards handle this)
        # Only penalize IMMEDIATE danger (dist <= 1), not general proximity
        for entity in self.entities:
            dist = abs(entity['pos'][0] - self.agent_pos[0]) +                    abs(entity['pos'][1] - self.agent_pos[1])
            if entity['danger'] > 0.5 and dist <= 1:
                reward -= 1.0  # Only immediate danger

        # PROXIMITY BONUS for being near rewards (balance with entity avoidance)
        if self.rewards:
            min_reward_dist = float('inf')
            for rx, ry in self.rewards:
                dist = abs(rx - self.agent_pos[0]) + abs(ry - self.agent_pos[1])
                min_reward_dist = min(min_reward_dist, dist)

            # Reward for being close to target (continuous signal)
            if min_reward_dist <= 2:
                reward += 1.0  # Very close to reward
            elif min_reward_dist <= 4:
                reward += 0.3  # Getting closer

        # Get temporal observation
        obs = self._get_observation()

        info.update({
            'steps': self.steps,
            'lives': self.lives,
            'rewards_left': len(self.rewards),
            'total_collected': self.total_collected
        })

        return obs, reward, done, info

    def _move(self, pos, action):
        """Move in direction, respecting walls"""
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x = pos[0] + dx
        new_y = pos[1] + dy

        if new_x < 0 or new_x >= self.grid_width or new_y < 0 or new_y >= self.grid_height:
            return pos

        if (new_x, new_y) in self.walls:
            return pos

        return (new_x, new_y)

    def _update_entities(self):
        """Update entity positions"""
        for entity in self.entities:
            if random.random() > self.entity_speed:
                continue

            behavior = entity['behavior']
            ex, ey = entity['pos']

            if behavior == 'chase':
                ax, ay = self.agent_pos
                dx = np.sign(ax - ex)
                dy = np.sign(ay - ey)

                if abs(ax - ex) > abs(ay - ey):
                    new_pos = self._move(entity['pos'], 3 if dx > 0 else 2)
                else:
                    new_pos = self._move(entity['pos'], 1 if dy > 0 else 0)

                entity['velocity'] = (new_pos[0] - ex, new_pos[1] - ey)
                entity['pos'] = new_pos

            elif behavior == 'patrol':
                action = random.randint(0, 3)
                new_pos = self._move(entity['pos'], action)
                entity['velocity'] = (new_pos[0] - ex, new_pos[1] - ey)
                entity['pos'] = new_pos

            elif behavior == 'random':
                if random.random() < 0.3:
                    action = random.randint(0, 3)
                    new_pos = self._move(entity['pos'], action)
                    entity['velocity'] = (new_pos[0] - ex, new_pos[1] - ey)
                    entity['pos'] = new_pos
                else:
                    entity['velocity'] = (0, 0)

    def _get_observation(self):
        """Get temporal flow observation"""
        world_state = {
            'agent_pos': self.agent_pos,
            'walls': self.walls,
            'entities': self.entities,
            'rewards': list(self.rewards),
            'grid_size': (self.grid_width, self.grid_height),
            'last_action': self.last_action
        }
        return self.observer.observe(world_state)

    def render_ascii(self):
        """ASCII rendering"""
        grid = [['.' for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        for wx, wy in self.walls:
            grid[wy][wx] = '#'

        for rx, ry in self.rewards:
            grid[ry][rx] = '*'

        for i, entity in enumerate(self.entities):
            ex, ey = entity['pos']
            grid[ey][ex] = str(i)

        ax, ay = self.agent_pos
        grid[ay][ax] = 'A'

        print("+" + "-" * self.grid_width + "+")
        for row in grid:
            print("|" + "".join(row) + "|")
        print("+" + "-" * self.grid_width + "+")


def demo():
    """Demo the temporal environment"""
    print("=" * 60)
    print("TEMPORAL RANDOM 2D ENVIRONMENT")
    print("Uses temporal flow observations for understanding")
    print("=" * 60)
    print()

    env = TemporalRandom2DEnv(
        grid_size=(20, 20),
        num_entities=3,
        num_rewards=10,  # Sparse rewards!
        maze_complexity=0.3,
        entity_speed=0.5
    )

    print(f"Observation dimension: {env.obs_dim}")
    print(f"  Current features: {env.observer.current_features}")
    print(f"  Delta features: {env.observer.delta_features}")
    print()

    obs = env.reset()
    print("Generated new maze:")
    env.render_ascii()
    print()

    print(f"Agent: A, Entities: 0-2, Rewards: * (only {len(env.rewards)}!)")
    print()

    # Check observation
    print("Observation analysis:")
    print(f"  Reward direction: ({obs[46]:.2f}, {obs[47]:.2f})")
    print(f"  Nearest entity distance: {obs[40]:.2f}")
    print(f"  Escape routes: {obs[36]:.2f}")
    print()

    # Take a few actions
    print("Taking 5 actions...")
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    for i in range(5):
        action = random.randint(0, 3)
        obs, reward, done, info = env.step(action)
        print(f"  {actions[action]}: reward={reward:.1f}, collected={info['collected_reward']}, died={info['died']}")
        print(f"    New reward direction: ({obs[46]:.2f}, {obs[47]:.2f})")
        if done:
            break
    print()

    print("Key improvements:")
    print("  1. Only 10 rewards (sparse) - forces target-seeking behavior")
    print("  2. Explicit reward direction in observation")
    print("  3. Temporal deltas show what's changing")
    print("  4. No noise padding - clean signal")


if __name__ == '__main__':
    demo()

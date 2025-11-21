"""
Temporal-Aware Observer for Foundation 2D Agent
Uses Latent Flow approach: current state + temporal delta = understanding

Philosophy: Agent must KNOW WHERE IT STANDS before making decisions.
- Current frame: "What do I see NOW?"
- Delta frame: "What CHANGED since last moment?"
- Combined: "I understand the situation"
"""
import numpy as np
import math


class TemporalFlowObserver:
    """
    Egocentric observer with explicit temporal flow encoding.

    Key difference from frame stacking:
    - Frame stacking: [raw_t-3, raw_t-2, raw_t-1, raw_t] - network must discover time
    - Temporal flow: [current_features, delta_features] - time is explicit

    Delta features tell the agent:
    - "Entity was far, now close" = approaching
    - "Reward was left, now right" = I moved
    - "Escape routes decreased" = getting trapped
    """

    def __init__(self, num_rays=8, ray_length=10):
        """
        Args:
            num_rays: Number of ray-cast directions
            ray_length: Max sensing distance
        """
        self.num_rays = num_rays
        self.ray_length = ray_length

        # Ray directions (8 cardinal + diagonal)
        self.ray_directions = [
            (0, -1),   # North (UP)
            (1, -1),   # Northeast
            (1, 0),    # East (RIGHT)
            (1, 1),    # Southeast
            (0, 1),    # South (DOWN)
            (-1, 1),   # Southwest
            (-1, 0),   # West (LEFT)
            (-1, -1),  # Northwest
        ]

        # Current frame features (what I see NOW)
        self.current_features = (
            num_rays * 3 +  # reward_dist, entity_dist, wall_dist per ray
            num_rays +      # danger levels per ray
            6 +             # topology (corridor, junction, dead_end, openness, escapes, density)
            8 +             # nearest entity info (rel_x, rel_y, dist, danger, approaching, count, avg_danger, convergence)
            2               # direction to nearest reward (dx, dy) - EXPLICIT TARGET SIGNAL
        )

        # Delta features (what CHANGED)
        self.delta_features = (
            num_rays * 3 +  # changes in reward/entity/wall distances
            6 +             # topology changes
            8 +             # entity movement changes
            2 +             # reward direction change
            4               # meta-deltas (danger_trend, escape_trend, progress_rate, entity_approach_rate)
        )

        # Total observation dimension
        self.obs_dim = self.current_features + self.delta_features

        # Memory of previous frame
        self.prev_frame = None
        self.prev_agent_pos = None

        # Spatial memory (persistent within episode)
        self.visited_positions = set()
        self.total_steps = 0

    def reset(self):
        """Reset for new episode"""
        self.prev_frame = None
        self.prev_agent_pos = None
        self.visited_positions = set()
        self.total_steps = 0

    def observe(self, world_state):
        """
        Generate observation: [current_features, delta_features]

        Args:
            world_state: dict with agent_pos, walls, entities, rewards, grid_size, last_action

        Returns:
            observation: numpy array with explicit temporal understanding
        """
        agent_pos = world_state['agent_pos']
        walls = world_state['walls']
        entities = world_state.get('entities', [])
        rewards = world_state.get('rewards', [])
        grid_size = world_state.get('grid_size', (20, 20))

        # === CURRENT FRAME FEATURES ===
        current = []

        # Ray-casting: what's around me NOW
        ray_features = self._cast_all_rays(agent_pos, walls, entities, rewards, grid_size)
        current.extend(ray_features['distances'])  # 8*3 = 24
        current.extend(ray_features['dangers'])    # 8

        # Topology: what kind of space am I in NOW
        topo = self._compute_topology(agent_pos, walls, grid_size)
        current.extend(topo)  # 6

        # Entity awareness: where are dangers NOW
        entity_info = self._compute_entity_info(agent_pos, entities)
        current.extend(entity_info)  # 8

        # EXPLICIT target direction: where is nearest reward NOW
        reward_dir = self._compute_reward_direction(agent_pos, rewards)
        current.extend(reward_dir)  # 2

        current = np.array(current, dtype=np.float32)

        # === DELTA FEATURES (temporal understanding) ===
        if self.prev_frame is None:
            # First observation - no history, zeros = "no change information"
            delta = np.zeros(self.delta_features, dtype=np.float32)
        else:
            delta = self._compute_deltas(current, self.prev_frame, agent_pos, entities)

        # Update memory
        self.prev_frame = current.copy()
        self.prev_agent_pos = agent_pos
        self.visited_positions.add(agent_pos)
        self.total_steps += 1

        # Combine: [what I see NOW, what CHANGED]
        observation = np.concatenate([current, delta])

        return observation

    def _cast_all_rays(self, agent_pos, walls, entities, rewards, grid_size):
        """Cast rays in all directions"""
        distances = []
        dangers = []

        for dx, dy in self.ray_directions:
            # Distance to reward in this direction
            reward_dist = self._ray_to_target(agent_pos, dx, dy, rewards, grid_size)
            distances.append(reward_dist)

            # Distance to entity in this direction
            entity_dist, danger = self._ray_to_entity(agent_pos, dx, dy, entities, grid_size)
            distances.append(entity_dist)
            dangers.append(danger)

            # Distance to wall in this direction
            wall_dist = self._ray_to_wall(agent_pos, dx, dy, walls, grid_size)
            distances.append(wall_dist)

        return {'distances': distances, 'dangers': dangers}

    def _ray_to_wall(self, agent_pos, dx, dy, walls, grid_size):
        """Distance to nearest wall in direction"""
        ax, ay = agent_pos
        for dist in range(1, self.ray_length + 1):
            check_x = ax + dx * dist
            check_y = ay + dy * dist

            # Out of bounds
            if check_x < 0 or check_x >= grid_size[0] or check_y < 0 or check_y >= grid_size[1]:
                return dist / self.ray_length

            # Wall hit
            if (check_x, check_y) in walls:
                return dist / self.ray_length

        return 1.0  # No wall within range

    def _ray_to_entity(self, agent_pos, dx, dy, entities, grid_size):
        """Distance to nearest entity in direction"""
        ax, ay = agent_pos
        min_dist = self.ray_length
        danger = 0.0

        for entity in entities:
            ex, ey = entity['pos']
            rel_x = ex - ax
            rel_y = ey - ay

            # Check if entity is in this ray direction
            if dx != 0 or dy != 0:
                ray_len = math.sqrt(dx*dx + dy*dy)
                dot = (rel_x * dx + rel_y * dy) / ray_len

                if dot > 0:  # In front
                    dist = math.sqrt(rel_x*rel_x + rel_y*rel_y)
                    if dist <= self.ray_length and dist > 0:
                        alignment = dot / dist
                        if alignment > 0.7:  # Within ~45 degree cone
                            if dist < min_dist:
                                min_dist = dist
                                danger = entity.get('danger', 0.5)

        return min_dist / self.ray_length, danger

    def _ray_to_target(self, agent_pos, dx, dy, rewards, grid_size):
        """Distance to nearest reward in direction"""
        ax, ay = agent_pos
        min_dist = self.ray_length

        for rx, ry in rewards:
            rel_x = rx - ax
            rel_y = ry - ay

            if dx != 0 or dy != 0:
                ray_len = math.sqrt(dx*dx + dy*dy)
                dot = (rel_x * dx + rel_y * dy) / ray_len

                if dot > 0:
                    dist = math.sqrt(rel_x*rel_x + rel_y*rel_y)
                    if dist <= self.ray_length and dist > 0:
                        alignment = dot / dist
                        if alignment > 0.7:
                            min_dist = min(min_dist, dist)

        return min_dist / self.ray_length

    def _compute_topology(self, agent_pos, walls, grid_size):
        """Local spatial topology"""
        ax, ay = agent_pos

        # Count open neighbors
        open_dirs = 0
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]:
                if (nx, ny) not in walls:
                    open_dirs += 1

        # Topology features
        is_corridor = float(open_dirs == 2)
        is_junction = float(open_dirs >= 3)
        is_dead_end = float(open_dirs == 1)

        # Local openness (3x3)
        open_tiles = 0
        total = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]:
                    total += 1
                    if (nx, ny) not in walls:
                        open_tiles += 1
        openness = open_tiles / max(total, 1)

        escape_routes = open_dirs / 4.0

        # Wall density (5x5)
        wall_count = 0
        total = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]:
                    total += 1
                    if (nx, ny) in walls:
                        wall_count += 1
        wall_density = wall_count / max(total, 1)

        return [is_corridor, is_junction, is_dead_end, openness, escape_routes, wall_density]

    def _compute_entity_info(self, agent_pos, entities):
        """Information about entities relative to agent"""
        if not entities:
            return [0.0] * 8

        ax, ay = agent_pos

        # Find nearest entity
        min_dist = float('inf')
        nearest = None
        for entity in entities:
            ex, ey = entity['pos']
            dist = abs(ex - ax) + abs(ey - ay)
            if dist < min_dist:
                min_dist = dist
                nearest = entity

        if nearest is None:
            return [0.0] * 8

        ex, ey = nearest['pos']
        rel_x = np.clip((ex - ax) / 10.0, -1, 1)
        rel_y = np.clip((ey - ay) / 10.0, -1, 1)
        dist_norm = min(min_dist / 20.0, 1.0)
        danger = nearest.get('danger', 0.5)

        # Is approaching?
        vx, vy = nearest.get('velocity', (0, 0))
        approaching = 0.0
        if vx != 0 or vy != 0:
            to_agent_x = ax - ex
            to_agent_y = ay - ey
            dot = vx * to_agent_x + vy * to_agent_y
            approaching = 1.0 if dot > 0 else 0.0

        # Nearby count
        nearby = sum(1 for e in entities if abs(e['pos'][0] - ax) + abs(e['pos'][1] - ay) <= 5)
        entity_count = min(nearby / 4.0, 1.0)

        avg_danger = sum(e.get('danger', 0.5) for e in entities) / len(entities)

        # Convergence
        convergence = 0.0
        if len(entities) >= 2:
            approaching_count = sum(
                1 for e in entities
                if (e.get('velocity', (0, 0))[0] * (ax - e['pos'][0]) +
                    e.get('velocity', (0, 0))[1] * (ay - e['pos'][1])) > 0
            )
            convergence = approaching_count / len(entities)

        return [rel_x, rel_y, dist_norm, danger, approaching, entity_count, avg_danger, convergence]

    def _compute_reward_direction(self, agent_pos, rewards):
        """
        EXPLICIT direction to nearest reward.
        This is the KEY FEATURE missing from original observer.

        Returns normalized (dx, dy) vector pointing to nearest reward.
        """
        if not rewards:
            return [0.0, 0.0]

        ax, ay = agent_pos

        # Find nearest reward
        min_dist = float('inf')
        nearest = None
        for rx, ry in rewards:
            dist = abs(rx - ax) + abs(ry - ay)
            if dist < min_dist:
                min_dist = dist
                nearest = (rx, ry)

        if nearest is None or min_dist == 0:
            return [0.0, 0.0]

        rx, ry = nearest

        # Direction vector (normalized to ~[-1, 1])
        dx = np.clip((rx - ax) / 10.0, -1, 1)
        dy = np.clip((ry - ay) / 10.0, -1, 1)

        return [dx, dy]

    def _compute_deltas(self, current, prev, agent_pos, entities):
        """
        Compute temporal deltas: what CHANGED between frames.

        This is where temporal understanding lives.
        """
        delta = []

        # Changes in ray distances (24 values: 8 rays * 3 distances)
        ray_deltas = current[:24] - prev[:24]
        delta.extend(ray_deltas)

        # Topology changes (6 values)
        topo_deltas = current[32:38] - prev[32:38]
        delta.extend(topo_deltas)

        # Entity info changes (8 values)
        entity_deltas = current[38:46] - prev[38:46]
        delta.extend(entity_deltas)

        # Reward direction change (2 values)
        reward_dir_delta = current[46:48] - prev[46:48]
        delta.extend(reward_dir_delta)

        # Meta-deltas (higher-level temporal patterns)

        # Danger trend: are entities getting closer overall?
        # If entity distances (indices 1, 4, 7, ... in ray data) decreased = danger increasing
        entity_dists_current = [current[i*3 + 1] for i in range(8)]
        entity_dists_prev = [prev[i*3 + 1] for i in range(8)]
        danger_trend = np.mean(entity_dists_prev) - np.mean(entity_dists_current)  # Positive = getting closer

        # Escape trend: are escape routes decreasing?
        escape_trend = current[36] - prev[36]  # escape_routes change

        # Progress rate: are we getting closer to rewards?
        reward_dists_current = [current[i*3] for i in range(8)]
        reward_dists_prev = [prev[i*3] for i in range(8)]
        progress_rate = np.mean(reward_dists_prev) - np.mean(reward_dists_current)  # Positive = getting closer

        # Entity approach rate: how fast are entities closing in?
        entity_approach_rate = current[42] - prev[42]  # approaching flag change

        delta.extend([danger_trend, escape_trend, progress_rate, entity_approach_rate])

        return np.array(delta, dtype=np.float32)


def demo():
    """Demonstrate temporal flow observation"""
    print("=" * 60)
    print("TEMPORAL FLOW OBSERVER")
    print("Understand WHERE YOU STAND before deciding")
    print("=" * 60)
    print()

    observer = TemporalFlowObserver()
    print(f"Observation dimension: {observer.obs_dim}")
    print(f"  Current features: {observer.current_features}")
    print(f"  Delta features: {observer.delta_features}")
    print()

    # Simulate agent moving while entity approaches
    world_state_t1 = {
        'agent_pos': (10, 10),
        'walls': {(8, 10), (12, 10)},
        'entities': [{'pos': (10, 15), 'velocity': (0, -1), 'danger': 0.8}],
        'rewards': [(10, 5)],  # Reward to the north
        'grid_size': (20, 20),
        'last_action': -1
    }

    print("Time step 1:")
    print(f"  Agent at: {world_state_t1['agent_pos']}")
    print(f"  Entity at: {world_state_t1['entities'][0]['pos']} (approaching)")
    print(f"  Reward at: {world_state_t1['rewards'][0]}")

    obs1 = observer.observe(world_state_t1)
    print(f"  Observation: {obs1.shape}")
    print(f"  Reward direction: ({obs1[46]:.2f}, {obs1[47]:.2f}) = North")
    print(f"  Delta features: all zeros (first frame)")
    print()

    # Agent moves north, entity follows
    world_state_t2 = {
        'agent_pos': (10, 9),  # Moved north
        'walls': {(8, 10), (12, 10)},
        'entities': [{'pos': (10, 14), 'velocity': (0, -1), 'danger': 0.8}],  # Closer
        'rewards': [(10, 5)],
        'grid_size': (20, 20),
        'last_action': 0  # UP
    }

    print("Time step 2:")
    print(f"  Agent moved to: {world_state_t2['agent_pos']} (north)")
    print(f"  Entity now at: {world_state_t2['entities'][0]['pos']} (followed)")

    obs2 = observer.observe(world_state_t2)
    print(f"  Observation: {obs2.shape}")
    print(f"  Reward direction: ({obs2[46]:.2f}, {obs2[47]:.2f})")

    # Extract deltas
    delta_start = observer.current_features
    danger_trend = obs2[delta_start + 24 + 6 + 8 + 2]
    progress_rate = obs2[delta_start + 24 + 6 + 8 + 2 + 2]

    print(f"  Danger trend: {danger_trend:.3f} (positive = entity closing)")
    print(f"  Progress rate: {progress_rate:.3f} (positive = reward closer)")
    print()

    print("KEY INSIGHT:")
    print("The agent now KNOWS:")
    print("  - Where the reward is (explicit direction)")
    print("  - That danger is approaching (delta trend)")
    print("  - That it made progress toward goal (progress rate)")
    print()
    print("With this understanding, it can DECIDE:")
    print("  - Continue toward reward? (progress good)")
    print("  - But danger increasing! (entity closing)")
    print("  - Maybe take evasive action first")


if __name__ == '__main__':
    demo()

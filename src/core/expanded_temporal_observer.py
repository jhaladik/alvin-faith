"""
Expanded Temporal Observer - Option A Implementation

Key Improvements:
1. Spatial Expansion: 16 rays (2x) with longer range (15 vs 10)
2. Multi-Scale Temporal: Micro (5) + Meso (20) + Macro (50) frames
3. Better pattern detection over longer timescales
4. Compatible with extended planning horizon (20-30 steps)

Philosophy: See farther in space, think longer in time
"""
import numpy as np
import math
from collections import deque


class ExpandedTemporalObserver:
    """
    Enhanced egocentric observer with expanded spatial and temporal capacity.

    Spatial expansion (2x coverage):
    - 16 rays vs 8 (more angular resolution)
    - Ray length 15 vs 10 (see farther)
    - Coverage: ~60% of grid vs ~25%

    Temporal expansion (multi-scale understanding):
    - Micro window: Last 5 frames (immediate)
    - Meso window: Last 20 frames (tactical patterns)
    - Macro window: Last 50 frames (strategic patterns)

    Observation breakdown:
    - Current features: ~80 dims (16 rays + global info)
    - Delta features: ~68 dims (immediate changes)
    - Multi-scale temporal: ~32 dims (pattern features)
    Total: ~180 dims (vs 92 dims baseline)
    """

    def __init__(self, num_rays=16, ray_length=15, verbose=False):
        """
        Args:
            num_rays: Number of ray-cast directions (default 16, was 8)
            ray_length: Max sensing distance (default 15, was 10)
            verbose: Print initialization info (default False to reduce log spam)
        """
        self.num_rays = num_rays
        self.ray_length = ray_length

        # Ray directions (16 directions for better angular resolution)
        angles = [i * (2 * math.pi / num_rays) for i in range(num_rays)]
        self.ray_directions = [
            (math.cos(angle), math.sin(angle))
            for angle in angles
        ]

        # Current frame features (what I see NOW)
        self.current_features = (
            num_rays * 3 +  # reward_dist, entity_dist, wall_dist per ray
            num_rays +      # danger levels per ray
            6 +             # topology (corridor, junction, dead_end, openness, escapes, density)
            8 +             # nearest entity info
            2               # direction to nearest reward
        )

        # Delta features (what CHANGED since last frame)
        self.delta_features = (
            num_rays * 3 +  # changes in reward/entity/wall distances
            6 +             # topology changes
            8 +             # entity movement changes
            2 +             # reward direction change
            4               # meta-deltas (danger_trend, escape_trend, progress_rate, entity_approach_rate)
        )

        # Multi-scale temporal features (NEW!)
        self.multi_scale_features = (
            8 +   # Micro pattern (5 frames): danger oscillation, movement consistency, trap detection
            8 +   # Meso pattern (20 frames): entity mode (chase/scatter), zone control, tactical position
            8 +   # Macro pattern (50 frames): strategic progress, exploration rate, long-term survival
            8     # Cross-scale: pattern stability, regime changes, uncertainty
        )

        # Total observation dimension
        self.obs_dim = self.current_features + self.delta_features + self.multi_scale_features

        if verbose:
            print(f"Expanded Temporal Observer initialized:")
            print(f"  Rays: {num_rays} (angular resolution: {360//num_rays}°)")
            print(f"  Ray length: {ray_length} tiles")
            print(f"  Current features: {self.current_features} dims")
            print(f"  Delta features: {self.delta_features} dims")
            print(f"  Multi-scale temporal: {self.multi_scale_features} dims")
            print(f"  Total observation: {self.obs_dim} dims")
            print(f"  Expansion: +{self.obs_dim - 92} dims vs baseline (92 dims)")

        # Multi-scale temporal buffers
        self.micro_buffer = deque(maxlen=5)    # Last 5 frames (immediate patterns)
        self.meso_buffer = deque(maxlen=20)    # Last 20 frames (tactical patterns)
        self.macro_buffer = deque(maxlen=50)   # Last 50 frames (strategic patterns)

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
        self.micro_buffer.clear()
        self.meso_buffer.clear()
        self.macro_buffer.clear()

    def observe(self, world_state):
        """
        Generate observation: [current_features, delta_features, multi_scale_features]

        Args:
            world_state: dict with agent_pos, walls, entities, rewards, grid_size

        Returns:
            observation: numpy array with expanded spatial-temporal understanding
        """
        agent_pos = world_state['agent_pos']
        walls = world_state['walls']
        entities = world_state.get('entities', [])
        rewards = world_state.get('rewards', [])
        grid_size = world_state.get('grid_size', (20, 20))

        self.total_steps += 1
        self.visited_positions.add(tuple(agent_pos))

        # Compute current frame features
        current_frame = self._compute_current_frame(
            agent_pos, walls, entities, rewards, grid_size
        )

        # Compute delta features (changes since last frame)
        if self.prev_frame is not None:
            delta_frame = self._compute_delta_frame(
                current_frame, self.prev_frame, agent_pos, self.prev_agent_pos
            )
        else:
            # First frame - no deltas available
            delta_frame = np.zeros(self.delta_features)

        # Add current frame info to temporal buffers
        frame_info = {
            'current': current_frame,
            'agent_pos': agent_pos,
            'entities': entities,
            'rewards': rewards,
            'danger_level': self._compute_danger_level(current_frame)
        }
        self.micro_buffer.append(frame_info)
        self.meso_buffer.append(frame_info)
        self.macro_buffer.append(frame_info)

        # Compute multi-scale temporal features
        multi_scale_frame = self._compute_multi_scale_features()

        # Update memory
        self.prev_frame = current_frame.copy()
        self.prev_agent_pos = agent_pos.copy() if isinstance(agent_pos, np.ndarray) else np.array(agent_pos)

        # Combine all features
        observation = np.concatenate([
            current_frame,      # What I see NOW
            delta_frame,        # What CHANGED
            multi_scale_frame   # Patterns over TIME
        ])

        assert len(observation) == self.obs_dim, \
            f"Observation size mismatch: {len(observation)} != {self.obs_dim}"

        return observation

    def _compute_current_frame(self, agent_pos, walls, entities, rewards, grid_size):
        """Compute current frame features with expanded spatial coverage"""
        features = []

        # 1. Ray-cast features (expanded to 16 rays)
        for direction in self.ray_directions:
            reward_dist, entity_dist, wall_dist = self._raycast(
                agent_pos, direction, walls, entities, rewards, grid_size
            )
            features.extend([reward_dist, entity_dist, wall_dist])

        # 2. Danger levels per ray
        for direction in self.ray_directions:
            danger = self._compute_ray_danger(
                agent_pos, direction, entities, grid_size
            )
            features.append(danger)

        # 3. Topology features
        topology = self._compute_topology(agent_pos, walls, grid_size)
        features.extend(topology)

        # 4. Nearest entity info (global)
        entity_info = self._compute_nearest_entity_info(agent_pos, entities)
        features.extend(entity_info)

        # 5. Direction to nearest reward (explicit target signal)
        reward_direction = self._compute_reward_direction(agent_pos, rewards)
        features.extend(reward_direction)

        return np.array(features, dtype=np.float32)

    def _raycast(self, agent_pos, direction, walls, entities, rewards, grid_size):
        """
        Cast a ray from agent position in given direction
        Returns: (reward_dist, entity_dist, wall_dist) normalized to [0,1]
        """
        ax, ay = agent_pos
        dx, dy = direction

        reward_dist = self.ray_length
        entity_dist = self.ray_length
        wall_dist = self.ray_length

        for step in range(1, self.ray_length + 1):
            # Current position along ray
            x = ax + dx * step
            y = ay + dy * step

            # Normalize to integer coordinates
            px, py = int(round(x)), int(round(y))

            # Check bounds
            if px < 0 or px >= grid_size[0] or py < 0 or py >= grid_size[1]:
                wall_dist = min(wall_dist, step)
                break

            # Check wall
            if (px, py) in walls:
                wall_dist = min(wall_dist, step)
                break

            # Check reward
            if rewards and reward_dist == self.ray_length:
                for reward_pos in rewards:
                    if abs(px - reward_pos[0]) < 1 and abs(py - reward_pos[1]) < 1:
                        reward_dist = min(reward_dist, step)

            # Check entity
            if entities and entity_dist == self.ray_length:
                for entity in entities:
                    ex, ey = entity['pos']
                    if abs(px - ex) < 1 and abs(py - ey) < 1:
                        entity_dist = min(entity_dist, step)

        # Normalize to [0,1]
        return (
            reward_dist / self.ray_length,
            entity_dist / self.ray_length,
            wall_dist / self.ray_length
        )

    def _compute_ray_danger(self, agent_pos, direction, entities, grid_size):
        """Compute danger level along a ray direction"""
        if not entities:
            return 0.0

        ax, ay = agent_pos
        dx, dy = direction

        max_danger = 0.0

        for step in range(1, self.ray_length + 1):
            x = ax + dx * step
            y = ay + dy * step
            px, py = int(round(x)), int(round(y))

            if px < 0 or px >= grid_size[0] or py < 0 or py >= grid_size[1]:
                break

            # Check entities at this position
            for entity in entities:
                ex, ey = entity['pos']
                dist = abs(px - ex) + abs(py - ey)

                if dist < 2:  # Entity nearby
                    danger = entity.get('danger', 1.0)
                    # Decay danger with distance along ray
                    danger_here = danger * (1.0 - step / self.ray_length)
                    max_danger = max(max_danger, danger_here)

        return max_danger

    def _compute_topology(self, agent_pos, walls, grid_size):
        """Compute local topology features"""
        ax, ay = agent_pos

        # Count walls in 3x3 neighborhood
        wall_count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (dx == 0 and dy == 0):
                    continue
                pos = (ax + dx, ay + dy)
                if pos in walls or pos[0] < 0 or pos[0] >= grid_size[0] or \
                   pos[1] < 0 or pos[1] >= grid_size[1]:
                    wall_count += 1

        # Topology classification
        is_corridor = (wall_count >= 5)  # 5+ walls = narrow passage
        is_junction = (wall_count <= 2)  # 2 or fewer walls = open junction
        is_dead_end = (wall_count >= 6)  # 6+ walls = dead end
        openness = (8 - wall_count) / 8.0  # How open is this position

        # Count escape routes (adjacent non-wall cells)
        escape_routes = 8 - wall_count

        # Local density (walls in 5x5 area)
        density_count = 0
        total_cells = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                pos = (ax + dx, ay + dy)
                total_cells += 1
                if pos in walls:
                    density_count += 1
        density = density_count / total_cells if total_cells > 0 else 0.0

        return [
            float(is_corridor),
            float(is_junction),
            float(is_dead_end),
            openness,
            escape_routes / 8.0,
            density
        ]

    def _compute_nearest_entity_info(self, agent_pos, entities):
        """Compute info about nearest entity"""
        if not entities:
            return [0.0] * 8

        ax, ay = agent_pos

        # Find nearest entity
        nearest = min(entities, key=lambda e: abs(e['pos'][0] - ax) + abs(e['pos'][1] - ay))
        ex, ey = nearest['pos']

        # Relative position
        rel_x = (ex - ax) / 20.0  # Normalize by grid size
        rel_y = (ey - ay) / 20.0

        # Distance
        dist = (abs(ex - ax) + abs(ey - ay)) / 20.0

        # Danger level
        danger = nearest.get('danger', 1.0)

        # Is entity approaching? (requires velocity)
        vx, vy = nearest.get('velocity', (0, 0))
        to_agent_x = ax - ex
        to_agent_y = ay - ey
        dot = vx * to_agent_x + vy * to_agent_y
        approaching = 1.0 if dot > 0 else 0.0

        # Entity count
        entity_count = len(entities) / 10.0  # Normalize (assume max 10 entities)

        # Average danger of all entities
        avg_danger = np.mean([e.get('danger', 1.0) for e in entities])

        # Convergence: are multiple entities closing in?
        approaching_count = sum(1 for e in entities
                               if e.get('velocity', (0, 0))[0] * (ax - e['pos'][0]) +
                                  e.get('velocity', (0, 0))[1] * (ay - e['pos'][1]) > 0)
        convergence = approaching_count / max(len(entities), 1)

        return [rel_x, rel_y, dist, danger, approaching, entity_count, avg_danger, convergence]

    def _compute_reward_direction(self, agent_pos, rewards):
        """Compute normalized direction to nearest reward"""
        if not rewards:
            return [0.0, 0.0]

        ax, ay = agent_pos

        # Find nearest reward
        nearest = min(rewards, key=lambda r: abs(r[0] - ax) + abs(r[1] - ay))
        rx, ry = nearest

        # Normalized direction vector
        dx = (rx - ax) / 20.0
        dy = (ry - ay) / 20.0

        return [dx, dy]

    def _compute_delta_frame(self, current, prev, agent_pos, prev_agent_pos):
        """Compute what changed since last frame"""
        # Ray distance deltas (reward_dist, entity_dist, wall_dist per ray)
        ray_deltas = current[:self.num_rays * 3] - prev[:self.num_rays * 3]

        # Topology deltas
        topo_start = self.num_rays * 4  # Skip ray features (3 dists + 1 danger per ray)
        topo_end = topo_start + 6
        topo_deltas = current[topo_start:topo_end] - prev[topo_start:topo_end]

        # Entity info deltas
        entity_start = topo_end
        entity_end = entity_start + 8
        entity_deltas = current[entity_start:entity_end] - prev[entity_start:entity_end]

        # Reward direction deltas
        reward_deltas = current[entity_end:entity_end+2] - prev[entity_end:entity_end+2]

        # Meta-deltas (second derivatives - how quickly things are changing)
        danger_trend = entity_deltas[3]  # How fast danger is increasing
        escape_trend = topo_deltas[4]     # How fast escape routes changing

        # Progress rate (how much did we move)
        agent_movement = np.linalg.norm(agent_pos - prev_agent_pos)
        progress_rate = agent_movement / 1.41  # Normalize by max diagonal distance

        # Entity approach rate (average of all entity distance changes)
        entity_approach_rate = np.mean(ray_deltas[1::3])  # Every 3rd element is entity_dist

        meta_deltas = [danger_trend, escape_trend, progress_rate, entity_approach_rate]

        return np.concatenate([
            ray_deltas,
            topo_deltas,
            entity_deltas,
            reward_deltas,
            meta_deltas
        ])

    def _compute_danger_level(self, current_frame):
        """Extract danger level from current frame"""
        # Entity info starts at: num_rays*4 + 6
        entity_start = self.num_rays * 4 + 6
        danger_idx = entity_start + 3  # danger is 4th element in entity_info
        return current_frame[danger_idx]

    def _compute_multi_scale_features(self):
        """
        NEW: Compute multi-scale temporal patterns

        This is the key innovation - understanding patterns at different timescales
        """
        features = []

        # 1. Micro pattern (last 5 frames) - immediate threats
        micro_features = self._compute_micro_patterns()
        features.extend(micro_features)

        # 2. Meso pattern (last 20 frames) - tactical patterns
        meso_features = self._compute_meso_patterns()
        features.extend(meso_features)

        # 3. Macro pattern (last 50 frames) - strategic trends
        macro_features = self._compute_macro_patterns()
        features.extend(macro_features)

        # 4. Cross-scale patterns - how patterns relate across timescales
        cross_features = self._compute_cross_scale_patterns()
        features.extend(cross_features)

        return np.array(features, dtype=np.float32)

    def _compute_micro_patterns(self):
        """Patterns in last 5 frames (immediate)"""
        if len(self.micro_buffer) < 2:
            return [0.0] * 8

        # Extract danger levels over micro window
        dangers = [frame['danger_level'] for frame in self.micro_buffer]

        # Danger oscillation (is danger fluctuating?)
        danger_variance = np.var(dangers) if len(dangers) > 1 else 0.0

        # Danger trend (increasing or decreasing?)
        danger_trend = dangers[-1] - dangers[0] if len(dangers) >= 2 else 0.0

        # Movement consistency (are we moving in same direction?)
        positions = [frame['agent_pos'] for frame in self.micro_buffer]
        if len(positions) >= 3:
            velocities = [
                (positions[i+1][0] - positions[i][0], positions[i+1][1] - positions[i][1])
                for i in range(len(positions) - 1)
            ]
            # Consistency = how similar are consecutive velocity vectors
            if len(velocities) >= 2:
                dot_products = [
                    velocities[i][0] * velocities[i+1][0] + velocities[i][1] * velocities[i+1][1]
                    for i in range(len(velocities) - 1)
                ]
                movement_consistency = np.mean(dot_products) if dot_products else 0.0
            else:
                movement_consistency = 0.0
        else:
            movement_consistency = 0.0

        # Trap detection (getting cornered?)
        # If danger increasing AND movement consistency low = potential trap
        potential_trap = float(danger_trend > 0.1 and movement_consistency < 0.0)

        # Escape success (danger decreasing AND moving consistently away)
        escape_success = float(danger_trend < -0.1 and movement_consistency > 0.5)

        # Entity count stability
        entity_counts = [len(frame['entities']) for frame in self.micro_buffer]
        entity_stability = 1.0 - np.std(entity_counts) / (np.mean(entity_counts) + 1e-6)

        # Reward availability (are rewards nearby?)
        reward_counts = [len(frame['rewards']) for frame in self.micro_buffer]
        avg_rewards = np.mean(reward_counts) / 10.0  # Normalize

        # Recent collision risk (danger spiked?)
        collision_risk = float(max(dangers) > 0.7 and danger_trend > 0.2)

        return [
            danger_variance,
            danger_trend,
            movement_consistency,
            potential_trap,
            escape_success,
            entity_stability,
            avg_rewards,
            collision_risk
        ]

    def _compute_meso_patterns(self):
        """Patterns in last 20 frames (tactical)"""
        if len(self.meso_buffer) < 5:
            return [0.0] * 8

        # Entity behavior mode detection (chase vs scatter vs random)
        # If entity consistently moves toward agent = CHASE
        # If entity moves away = SCATTER
        # If random = RANDOM

        positions = [frame['agent_pos'] for frame in self.meso_buffer]
        entities_history = [frame['entities'] for frame in self.meso_buffer]

        # Track first entity (if exists)
        if entities_history[0]:
            chase_count = 0
            scatter_count = 0
            total_count = 0

            for i in range(1, len(entities_history)):
                if not entities_history[i] or not entities_history[i-1]:
                    continue

                # Compare distances
                prev_dist = abs(entities_history[i-1][0]['pos'][0] - positions[i-1][0]) + \
                           abs(entities_history[i-1][0]['pos'][1] - positions[i-1][1])
                curr_dist = abs(entities_history[i][0]['pos'][0] - positions[i][0]) + \
                           abs(entities_history[i][0]['pos'][1] - positions[i][1])

                if curr_dist < prev_dist:
                    chase_count += 1
                elif curr_dist > prev_dist:
                    scatter_count += 1
                total_count += 1

            if total_count > 0:
                chase_mode = chase_count / total_count
                scatter_mode = scatter_count / total_count
                random_mode = 1.0 - chase_mode - scatter_mode
            else:
                chase_mode = scatter_mode = random_mode = 0.0
        else:
            chase_mode = scatter_mode = random_mode = 0.0

        # Zone control (how much of grid explored in meso window?)
        unique_positions = set(tuple(frame['agent_pos']) for frame in self.meso_buffer)
        zone_coverage = len(unique_positions) / min(len(self.meso_buffer), 400)  # Normalize by 20x20

        # Tactical position (am I in good position tactically?)
        # Good = low danger, multiple escape routes, rewards available
        dangers = [frame['danger_level'] for frame in self.meso_buffer]
        tactical_quality = 1.0 - np.mean(dangers)

        # Survival time in window (how long since last danger spike?)
        frames_since_danger = 0
        for i in range(len(dangers) - 1, -1, -1):
            if dangers[i] > 0.7:
                break
            frames_since_danger += 1
        survival_stability = frames_since_danger / len(dangers)

        # Reward collection rate
        reward_counts = [len(frame['rewards']) for frame in self.meso_buffer]
        if len(reward_counts) >= 2:
            reward_delta = reward_counts[0] - reward_counts[-1]  # How many collected
            collection_rate = max(0, reward_delta) / len(reward_counts)
        else:
            collection_rate = 0.0

        # Evasion skill (successfully avoiding entities?)
        evasion_skill = scatter_mode + random_mode * 0.5  # Higher if not being chased

        return [
            chase_mode,
            scatter_mode,
            random_mode,
            zone_coverage,
            tactical_quality,
            survival_stability,
            collection_rate,
            evasion_skill
        ]

    def _compute_macro_patterns(self):
        """Patterns in last 50 frames (strategic)"""
        if len(self.macro_buffer) < 10:
            return [0.0] * 8

        # Strategic progress (are we making overall progress?)
        reward_counts = [len(frame['rewards']) for frame in self.macro_buffer]
        if len(reward_counts) >= 2:
            total_collected = reward_counts[0] - reward_counts[-1]
            progress_rate = max(0, total_collected) / len(reward_counts)
        else:
            progress_rate = 0.0

        # Exploration rate (how much of grid explored?)
        all_positions = set()
        for frame in self.macro_buffer:
            all_positions.add(tuple(frame['agent_pos']))
        exploration_coverage = len(all_positions) / 400  # Normalize by 20x20 grid

        # Long-term survival (average danger level)
        dangers = [frame['danger_level'] for frame in self.macro_buffer]
        avg_danger = np.mean(dangers)
        survival_quality = 1.0 - avg_danger

        # Danger exposure time (what % of time in danger?)
        danger_time = sum(1 for d in dangers if d > 0.5) / len(dangers)

        # Strategic cycling (are we revisiting same areas?)
        position_visits = {}
        for frame in self.macro_buffer:
            pos = tuple(frame['agent_pos'])
            position_visits[pos] = position_visits.get(pos, 0) + 1
        max_visits = max(position_visits.values()) if position_visits else 1
        cycling_behavior = max_visits / len(self.macro_buffer)

        # Learning progress (is danger decreasing over time?)
        if len(dangers) >= 20:
            early_danger = np.mean(dangers[:len(dangers)//2])
            late_danger = np.mean(dangers[len(dangers)//2:])
            learning_signal = max(0, early_danger - late_danger)
        else:
            learning_signal = 0.0

        # Efficiency (progress per step)
        efficiency = progress_rate / max(danger_time, 0.01)

        # Stamina (can we maintain performance?)
        # Check if performance declining in later frames
        if len(reward_counts) >= 20:
            early_rate = (reward_counts[0] - reward_counts[len(reward_counts)//2]) / (len(reward_counts)//2)
            late_rate = (reward_counts[len(reward_counts)//2] - reward_counts[-1]) / (len(reward_counts)//2)
            stamina = 1.0 if late_rate >= early_rate * 0.8 else 0.5
        else:
            stamina = 1.0

        return [
            progress_rate,
            exploration_coverage,
            survival_quality,
            danger_time,
            cycling_behavior,
            learning_signal,
            efficiency,
            stamina
        ]

    def _compute_cross_scale_patterns(self):
        """How do patterns relate across timescales?"""
        if len(self.micro_buffer) < 2 or len(self.meso_buffer) < 5 or len(self.macro_buffer) < 10:
            return [0.0] * 8

        # Pattern stability (are micro patterns consistent with meso/macro?)
        micro = self._compute_micro_patterns()
        meso = self._compute_meso_patterns()
        macro = self._compute_macro_patterns()

        # Micro-meso alignment
        micro_danger_trend = micro[1]  # Danger trend in micro
        meso_tactical_quality = meso[4]  # Tactical quality in meso
        micro_meso_align = 1.0 - abs(micro_danger_trend - (1.0 - meso_tactical_quality))

        # Meso-macro alignment
        meso_collection = meso[6]  # Collection rate in meso
        macro_progress = macro[0]  # Progress rate in macro
        meso_macro_align = 1.0 - abs(meso_collection - macro_progress)

        # Regime change detection (sudden shift in patterns)
        # If micro shows danger but meso/macro show safety = regime change!
        regime_change = float(micro[1] > 0.3 and meso[4] > 0.7)  # Danger spike in safe period

        # Uncertainty (high variance across scales = uncertain)
        danger_levels = [micro[0], 1.0 - meso[4], 1.0 - macro[2]]
        uncertainty = np.var(danger_levels)

        # Prediction confidence (how well does macro predict micro?)
        # If macro shows progress and micro shows success = high confidence
        prediction_confidence = float(macro[0] > 0.1 and micro[4] > 0.5)

        # Adaptation speed (how quickly do patterns change across scales?)
        # Fast = micro != meso != macro
        # Slow = all aligned
        adaptation_speed = abs(micro[1] - meso[1]) + abs(meso[1] - macro[0])

        # Strategic coherence (are all scales pointing same direction?)
        coherence = (micro_meso_align + meso_macro_align) / 2.0

        # Risk level (combination of immediate danger and long-term exposure)
        risk_level = micro[7] * 0.5 + macro[3] * 0.5  # Collision risk + danger exposure

        return [
            micro_meso_align,
            meso_macro_align,
            regime_change,
            uncertainty,
            prediction_confidence,
            adaptation_speed,
            coherence,
            risk_level
        ]


if __name__ == '__main__':
    # Test the expanded observer
    print("="*60)
    print("Testing Expanded Temporal Observer")
    print("="*60)

    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)

    # Create dummy world state
    world_state = {
        'agent_pos': np.array([10, 10]),
        'walls': set([(5, 5), (5, 6), (5, 7)]),
        'entities': [
            {'pos': (8, 8), 'danger': 1.0, 'velocity': (1, 1)}
        ],
        'rewards': [(12, 12), (15, 15)],
        'grid_size': (20, 20)
    }

    # Test observation
    obs = observer.observe(world_state)
    print(f"\nObservation shape: {obs.shape}")
    print(f"Expected shape: ({observer.obs_dim},)")
    print(f"Match: {obs.shape[0] == observer.obs_dim}")

    # Test multiple frames
    print("\nTesting multi-frame observation...")
    for i in range(10):
        world_state['agent_pos'] = np.array([10 + i, 10])
        obs = observer.observe(world_state)
        print(f"Frame {i+1}: obs shape = {obs.shape}, micro buffer = {len(observer.micro_buffer)}")

    print("\nExpanded Temporal Observer test complete!")

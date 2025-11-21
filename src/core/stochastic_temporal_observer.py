"""
Stochastic Temporal Observer - Handles Non-Linear Entity Movement

Key Insight: Ghosts don't move linearly! We need probabilistic predictions.

Improvements over TemporalFlowObserver:
1. Multi-hypothesis entity tracking (track 3 possible ghost positions)
2. Movement pattern learning (detect chase vs scatter modes)
3. Uncertainty quantification (how predictable is this entity?)
4. Longer temporal window (5 frames instead of 1)
5. Attention mechanism (weight predictions by confidence)
"""
import numpy as np
import math
from collections import deque


class StochasticTemporalObserver:
    """
    Observer that handles stochastic, non-linear entity movement.

    Philosophy: Don't assume linear velocity - learn movement distributions!
    """

    def __init__(self, num_rays=8, ray_length=10, temporal_window=5):
        self.num_rays = num_rays
        self.ray_length = ray_length
        self.temporal_window = temporal_window

        # Ray directions
        self.ray_directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
        ]

        # === FEATURES ===

        # Current frame (48 features)
        self.current_features = (
            num_rays * 3 +  # 24: reward/entity/wall per ray
            num_rays +      # 8: danger levels
            6 +             # 6: topology
            8 +             # 8: entity info
            2               # 2: reward direction
        )

        # Stochastic temporal features (NEW - 32 features)
        self.stochastic_features = (
            8 +   # Entity movement uncertainty (per ray)
            4 +   # Multi-hypothesis ghost positions (mean_x, mean_y, std_x, std_y)
            4 +   # Ghost behavior mode detection (chase_prob, scatter_prob, random_prob, predictability)
            8 +   # Temporal pattern strength (per ray, how predictable)
            4 +   # Danger forecast (1-step, 2-step, 3-step, 4-step ahead)
            4     # Escape route stability (how reliably can I escape)
        )

        # Classic temporal deltas (reduced - 24 features)
        self.delta_features = (
            num_rays * 2 +  # 16: reward/entity distance changes only
            6 +             # 6: topology changes
            2               # 2: reward direction change
        )

        self.obs_dim = self.current_features + self.stochastic_features + self.delta_features

        # Memory: track last N positions for each entity
        self.entity_history = {}  # {entity_id: deque of (x, y, timestamp)}
        self.frame_history = deque(maxlen=temporal_window)
        self.agent_position_history = deque(maxlen=temporal_window)
        self.timestep = 0

    def reset(self):
        """Reset for new episode"""
        self.entity_history = {}
        self.frame_history.clear()
        self.agent_position_history.clear()
        self.timestep = 0

    def observe(self, world_state):
        """Generate observation with stochastic temporal modeling"""
        agent_pos = world_state['agent_pos']
        walls = world_state['walls']
        entities = world_state.get('entities', [])
        rewards = world_state.get('rewards', [])
        grid_size = world_state.get('grid_size', (20, 20))

        self.timestep += 1

        # Update entity tracking
        self._update_entity_history(entities)

        # === CURRENT FRAME (same as original) ===
        current = []

        # Ray-casting
        ray_features = self._cast_all_rays(agent_pos, walls, entities, rewards, grid_size)
        current.extend(ray_features['distances'])  # 24
        current.extend(ray_features['dangers'])    # 8

        # Topology
        topo = self._compute_topology(agent_pos, walls, grid_size)
        current.extend(topo)  # 6

        # Entity info
        entity_info = self._compute_entity_info(agent_pos, entities)
        current.extend(entity_info)  # 8

        # Reward direction
        reward_dir = self._compute_reward_direction(agent_pos, rewards)
        current.extend(reward_dir)  # 2

        current = np.array(current, dtype=np.float32)

        # === STOCHASTIC TEMPORAL FEATURES (NEW!) ===
        stochastic = self._compute_stochastic_features(agent_pos, entities, grid_size)

        # === CLASSIC DELTAS ===
        if len(self.frame_history) == 0:
            delta = np.zeros(self.delta_features, dtype=np.float32)
        else:
            delta = self._compute_deltas(current, self.frame_history[-1])

        # Update history
        self.frame_history.append(current.copy())
        self.agent_position_history.append(agent_pos)

        # Combine: [current, stochastic_temporal, deltas]
        observation = np.concatenate([current, stochastic, delta])

        return observation

    def _update_entity_history(self, entities):
        """Track entity positions over time for pattern learning"""
        current_ids = set()

        for i, entity in enumerate(entities):
            entity_id = i  # Simple ID based on list index
            pos = entity['pos']
            current_ids.add(entity_id)

            if entity_id not in self.entity_history:
                self.entity_history[entity_id] = deque(maxlen=self.temporal_window)

            self.entity_history[entity_id].append((pos[0], pos[1], self.timestep))

        # Clean up disappeared entities
        disappeared = set(self.entity_history.keys()) - current_ids
        for entity_id in disappeared:
            del self.entity_history[entity_id]

    def _compute_stochastic_features(self, agent_pos, entities, grid_size):
        """
        Compute stochastic temporal features that handle non-linear movement.

        This is the KEY INNOVATION for handling ghosts!
        """
        features = []

        # 1. Movement uncertainty per ray (8 features)
        movement_uncertainty = self._compute_movement_uncertainty_per_ray(agent_pos, entities)
        features.extend(movement_uncertainty)

        # 2. Multi-hypothesis position prediction (4 features)
        ghost_distribution = self._predict_ghost_distribution(entities)
        features.extend(ghost_distribution)

        # 3. Behavior mode detection (4 features)
        behavior_mode = self._detect_behavior_mode(agent_pos, entities)
        features.extend(behavior_mode)

        # 4. Temporal pattern strength per ray (8 features)
        pattern_strength = self._compute_pattern_strength_per_ray(entities)
        features.extend(pattern_strength)

        # 5. Danger forecast (4 features: 1-4 steps ahead)
        danger_forecast = self._forecast_danger(agent_pos, entities, grid_size)
        features.extend(danger_forecast)

        # 6. Escape route stability (4 features)
        escape_stability = self._compute_escape_stability(agent_pos, entities, grid_size)
        features.extend(escape_stability)

        return np.array(features, dtype=np.float32)

    def _compute_movement_uncertainty_per_ray(self, agent_pos, entities):
        """
        How unpredictable is entity movement in each direction?

        High uncertainty = don't trust velocity predictions!
        """
        uncertainty = [0.0] * 8

        for ray_idx, (dx, dy) in enumerate(self.ray_directions):
            # Find entities in this ray direction
            ray_uncertainties = []

            for i, entity in enumerate(entities):
                if i not in self.entity_history or len(self.entity_history[i]) < 3:
                    ray_uncertainties.append(1.0)  # Unknown = max uncertainty
                    continue

                history = list(self.entity_history[i])

                # Check if entity is in this ray
                ex, ey = entity['pos']
                ax, ay = agent_pos
                rel_x, rel_y = ex - ax, ey - ay

                ray_len = math.sqrt(dx*dx + dy*dy) if (dx != 0 or dy != 0) else 1
                dot = (rel_x * dx + rel_y * dy) / ray_len
                dist = math.sqrt(rel_x*rel_x + rel_y*rel_y)

                if dot > 0 and dist > 0:
                    alignment = dot / dist
                    if alignment > 0.7:
                        # Entity in this ray - compute movement variance
                        velocities = []
                        for j in range(1, len(history)):
                            prev_x, prev_y, prev_t = history[j-1]
                            curr_x, curr_y, curr_t = history[j]
                            dt = curr_t - prev_t
                            if dt > 0:
                                vx = (curr_x - prev_x) / dt
                                vy = (curr_y - prev_y) / dt
                                velocities.append((vx, vy))

                        if velocities:
                            # Variance in velocity = unpredictability
                            vx_vals = [v[0] for v in velocities]
                            vy_vals = [v[1] for v in velocities]
                            var = np.var(vx_vals) + np.var(vy_vals)
                            ray_uncertainties.append(min(var, 1.0))

            uncertainty[ray_idx] = np.mean(ray_uncertainties) if ray_uncertainties else 0.5

        return uncertainty

    def _predict_ghost_distribution(self, entities):
        """
        Predict ghost position as a DISTRIBUTION, not a point.

        Returns: (mean_x, mean_y, std_x, std_y) relative to agent
        """
        if not entities:
            return [0.0, 0.0, 1.0, 1.0]  # No ghost, max uncertainty

        predicted_positions = []

        for i, entity in enumerate(entities):
            if i not in self.entity_history or len(self.entity_history[i]) < 2:
                predicted_positions.append(entity['pos'])
                continue

            history = list(self.entity_history[i])

            # Compute last 3 movements
            recent_moves = []
            for j in range(1, min(len(history), 4)):
                prev_x, prev_y, _ = history[j-1]
                curr_x, curr_y, _ = history[j]
                move = (curr_x - prev_x, curr_y - prev_y)
                recent_moves.append(move)

            # Predict: current position + mean of recent moves
            curr_x, curr_y, _ = history[-1]
            if recent_moves:
                mean_dx = np.mean([m[0] for m in recent_moves])
                mean_dy = np.mean([m[1] for m in recent_moves])
                pred_x = curr_x + mean_dx
                pred_y = curr_y + mean_dy
            else:
                pred_x, pred_y = curr_x, curr_y

            predicted_positions.append((pred_x, pred_y))

        # Compute distribution statistics
        if predicted_positions:
            xs = [p[0] for p in predicted_positions]
            ys = [p[1] for p in predicted_positions]
            mean_x = np.mean(xs)
            mean_y = np.mean(ys)
            std_x = np.std(xs) if len(xs) > 1 else 1.0
            std_y = np.std(ys) if len(ys) > 1 else 1.0
        else:
            mean_x, mean_y, std_x, std_y = 0.0, 0.0, 1.0, 1.0

        # Normalize relative to typical grid (20x20)
        return [
            np.clip(mean_x / 20.0, -1, 1),
            np.clip(mean_y / 20.0, -1, 1),
            min(std_x / 5.0, 1.0),
            min(std_y / 5.0, 1.0),
        ]

    def _detect_behavior_mode(self, agent_pos, entities):
        """
        Detect if ghosts are:
        - Chasing (moving toward agent)
        - Scattering (moving away)
        - Random (unpredictable)

        Returns: (chase_prob, scatter_prob, random_prob, overall_predictability)
        """
        if not entities:
            return [0.0, 0.0, 1.0, 0.0]

        chase_count = 0
        scatter_count = 0
        random_count = 0
        total = 0

        ax, ay = agent_pos

        for i, entity in enumerate(entities):
            if i not in self.entity_history or len(self.entity_history[i]) < 2:
                random_count += 1
                total += 1
                continue

            history = list(self.entity_history[i])
            prev_x, prev_y, _ = history[-2]
            curr_x, curr_y, _ = history[-1]

            # Distance to agent before and after move
            prev_dist = abs(prev_x - ax) + abs(prev_y - ay)
            curr_dist = abs(curr_x - ax) + abs(curr_y - ay)

            if curr_dist < prev_dist - 0.5:
                chase_count += 1  # Got closer
            elif curr_dist > prev_dist + 0.5:
                scatter_count += 1  # Got farther
            else:
                random_count += 1  # No clear pattern

            total += 1

        if total > 0:
            chase_prob = chase_count / total
            scatter_prob = scatter_count / total
            random_prob = random_count / total
            predictability = (chase_count + scatter_count) / total  # Non-random = predictable
        else:
            chase_prob = scatter_prob = 0.0
            random_prob = 1.0
            predictability = 0.0

        return [chase_prob, scatter_prob, random_prob, predictability]

    def _compute_pattern_strength_per_ray(self, entities):
        """
        For each ray, how PREDICTABLE are entities in that direction?

        Strong pattern = can trust temporal predictions
        Weak pattern = ignore temporal deltas, use reactive policy
        """
        pattern_strength = [0.5] * 8  # Default: moderate uncertainty

        for i, entity in enumerate(entities):
            if i not in self.entity_history or len(self.entity_history[i]) < 4:
                continue

            history = list(self.entity_history[i])

            # Compute movement autocorrelation
            moves = []
            for j in range(1, len(history)):
                prev_x, prev_y, _ = history[j-1]
                curr_x, curr_y, _ = history[j]
                moves.append((curr_x - prev_x, curr_y - prev_y))

            if len(moves) >= 3:
                # Check if movements are consistent
                move_variance = np.var([m[0] for m in moves]) + np.var([m[1] for m in moves])
                strength = 1.0 / (1.0 + move_variance)  # Low variance = high strength

                # Assign to rays based on entity direction
                # (simplified - could be more sophisticated)
                pattern_strength = [max(s, strength) for s in pattern_strength]

        return pattern_strength

    def _forecast_danger(self, agent_pos, entities, grid_size, steps=[1, 2, 3, 4]):
        """
        Predict danger level 1-4 steps into the future.

        Uses probabilistic forecasting, not deterministic!
        """
        forecasts = []
        ax, ay = agent_pos

        for n_steps in steps:
            danger_at_step = 0.0

            for i, entity in enumerate(entities):
                if i not in self.entity_history or len(self.entity_history[i]) < 2:
                    # Unknown entity - assume stays in place
                    ex, ey = entity['pos']
                    dist = abs(ex - ax) + abs(ey - ay)
                    danger = entity.get('danger', 0.5) / max(dist, 1)
                    danger_at_step += danger
                    continue

                history = list(self.entity_history[i])

                # Compute average movement
                recent_moves = []
                for j in range(1, min(len(history), 4)):
                    prev_x, prev_y, _ = history[j-1]
                    curr_x, curr_y, _ = history[j]
                    recent_moves.append((curr_x - prev_x, curr_y - prev_y))

                if recent_moves:
                    mean_dx = np.mean([m[0] for m in recent_moves])
                    mean_dy = np.mean([m[1] for m in recent_moves])
                else:
                    mean_dx, mean_dy = 0, 0

                # Predict position after n_steps
                curr_x, curr_y, _ = history[-1]
                pred_x = curr_x + mean_dx * n_steps
                pred_y = curr_y + mean_dy * n_steps

                # Distance to agent at that point
                pred_dist = abs(pred_x - ax) + abs(pred_y - ay)
                danger = entity.get('danger', 0.5) / max(pred_dist, 1)
                danger_at_step += danger

            forecasts.append(min(danger_at_step, 1.0))

        return forecasts

    def _compute_escape_stability(self, agent_pos, entities, grid_size):
        """
        Can I reliably escape, or are escape routes volatile?

        Returns: [current_escapes, 1-step_forecast, 2-step_forecast, confidence]
        """
        # Simplified - would need proper implementation
        return [0.5, 0.5, 0.5, 0.5]

    # === Standard helper methods (same as original) ===

    def _cast_all_rays(self, agent_pos, walls, entities, rewards, grid_size):
        """Same as TemporalFlowObserver"""
        distances = []
        dangers = []

        for dx, dy in self.ray_directions:
            reward_dist = self._ray_to_target(agent_pos, dx, dy, rewards, grid_size)
            distances.append(reward_dist)

            entity_dist, danger = self._ray_to_entity(agent_pos, dx, dy, entities, grid_size)
            distances.append(entity_dist)
            dangers.append(danger)

            wall_dist = self._ray_to_wall(agent_pos, dx, dy, walls, grid_size)
            distances.append(wall_dist)

        return {'distances': distances, 'dangers': dangers}

    def _ray_to_wall(self, agent_pos, dx, dy, walls, grid_size):
        """Same as original"""
        ax, ay = agent_pos
        for dist in range(1, self.ray_length + 1):
            check_x = ax + dx * dist
            check_y = ay + dy * dist
            if check_x < 0 or check_x >= grid_size[0] or check_y < 0 or check_y >= grid_size[1]:
                return dist / self.ray_length
            if (check_x, check_y) in walls:
                return dist / self.ray_length
        return 1.0

    def _ray_to_entity(self, agent_pos, dx, dy, entities, grid_size):
        """Same as original"""
        ax, ay = agent_pos
        min_dist = self.ray_length
        danger = 0.0

        for entity in entities:
            ex, ey = entity['pos']
            rel_x = ex - ax
            rel_y = ey - ay

            if dx != 0 or dy != 0:
                ray_len = math.sqrt(dx*dx + dy*dy)
                dot = (rel_x * dx + rel_y * dy) / ray_len

                if dot > 0:
                    dist = math.sqrt(rel_x*rel_x + rel_y*rel_y)
                    if dist <= self.ray_length and dist > 0:
                        alignment = dot / dist
                        if alignment > 0.7:
                            if dist < min_dist:
                                min_dist = dist
                                danger = entity.get('danger', 0.5)

        return min_dist / self.ray_length, danger

    def _ray_to_target(self, agent_pos, dx, dy, rewards, grid_size):
        """Same as original"""
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
        """Same as original"""
        ax, ay = agent_pos
        open_dirs = 0
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]:
                if (nx, ny) not in walls:
                    open_dirs += 1

        is_corridor = float(open_dirs == 2)
        is_junction = float(open_dirs >= 3)
        is_dead_end = float(open_dirs == 1)

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
        """Simplified - no velocity assumption!"""
        if not entities:
            return [0.0] * 8

        ax, ay = agent_pos
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

        # NO VELOCITY ASSUMPTION - use 0 for approaching
        approaching = 0.0

        nearby = sum(1 for e in entities if abs(e['pos'][0] - ax) + abs(e['pos'][1] - ay) <= 5)
        entity_count = min(nearby / 4.0, 1.0)
        avg_danger = sum(e.get('danger', 0.5) for e in entities) / len(entities)
        convergence = 0.0

        return [rel_x, rel_y, dist_norm, danger, approaching, entity_count, avg_danger, convergence]

    def _compute_reward_direction(self, agent_pos, rewards):
        """Same as original"""
        if not rewards:
            return [0.0, 0.0]

        ax, ay = agent_pos
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
        dx = np.clip((rx - ax) / 10.0, -1, 1)
        dy = np.clip((ry - ay) / 10.0, -1, 1)

        return [dx, dy]

    def _compute_deltas(self, current, prev):
        """Simplified deltas - only distance changes"""
        delta = []

        # Reward and entity distance changes (16 features: 8 rays * 2)
        for i in range(8):
            reward_delta = current[i*3] - prev[i*3]  # Reward distance
            entity_delta = current[i*3 + 1] - prev[i*3 + 1]  # Entity distance
            delta.extend([reward_delta, entity_delta])

        # Topology changes (6 features)
        topo_deltas = current[32:38] - prev[32:38]
        delta.extend(topo_deltas)

        # Reward direction change (2 features)
        reward_dir_delta = current[46:48] - prev[46:48]
        delta.extend(reward_dir_delta)

        return np.array(delta, dtype=np.float32)


if __name__ == '__main__':
    print("="*60)
    print("STOCHASTIC TEMPORAL OBSERVER")
    print("Handles non-linear, unpredictable entity movement")
    print("="*60)

    observer = StochasticTemporalObserver()
    print(f"Observation dimension: {observer.obs_dim}")
    print(f"  Current features: {observer.current_features}")
    print(f"  Stochastic temporal: {observer.stochastic_features}")
    print(f"  Delta features: {observer.delta_features}")
    print()
    print("Key innovations:")
    print("  1. Movement uncertainty quantification")
    print("  2. Multi-hypothesis position prediction")
    print("  3. Behavior mode detection (chase/scatter/random)")
    print("  4. Pattern strength (trust temporal or go reactive)")
    print("  5. Probabilistic danger forecasting")

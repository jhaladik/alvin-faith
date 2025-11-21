"""
Visual Test Games for Faith-Based Evolutionary Agent
Watch the agent play Snake, Pac-Man, and Dungeon with revolutionary capabilities:
- Faith pattern evolution (persistent exploration)
- Entity discovery (learns what entities are)
- Universal pattern transfer (game-agnostic strategies)
- Mechanic hypothesis testing (discovers hidden rules)
"""
import pygame
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

from core.planning_test_games import SnakeGame, PacManGame, DungeonGame
from context_aware_agent import (
    ContextAwareDQN,
    infer_context_from_observation,
    add_context_to_observation
)
from core.temporal_observer import TemporalFlowObserver
from core.world_model import WorldModelNetwork

# Import faith system modules
from core.faith_system import FaithPattern, FaithPopulation
from core.entity_discovery import EntityDiscoveryWorldModel, EntityBehaviorLearner
from core.pattern_transfer import UniversalPatternExtractor
from core.mechanic_detectors import MechanicDetector

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
GRAY = (128, 128, 128)
DARK_GREEN = (0, 100, 0)
GOLD = (255, 215, 0)
PURPLE = (128, 0, 128)
MAGENTA = (255, 0, 255)
TEAL = (0, 128, 128)


class FaithVisualRunner:
    """Run games with faith-based evolutionary agent and visual rendering"""

    def __init__(self, model_path, cell_size=25, use_planning=True, planning_freq=0.3,
                 planning_horizon=5, faith_freq=0.1):
        self.cell_size = cell_size
        self.use_planning = use_planning
        self.planning_freq = planning_freq
        self.planning_horizon = planning_horizon
        self.faith_freq = faith_freq

        # Load faith-based agent
        self.agent = None
        self.world_model = None
        self.entity_world_model = None
        self.observer = TemporalFlowObserver()
        self.current_context = None
        self.current_context_name = "Unknown"

        # Faith system components
        self.faith_population = None
        self.pattern_extractor = UniversalPatternExtractor()
        self.mechanic_detector = MechanicDetector()
        self.entity_learner = EntityBehaviorLearner()

        # Tracking stats
        self.faith_discoveries = []
        self.entity_types_detected = []
        self.patterns_detected = {}
        self.mechanics_confirmed = {}
        self.action_counts = {'faith': 0, 'planning': 0, 'reactive': 0}

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.agent = ContextAwareDQN(obs_dim=95, action_dim=4)
            self.agent.load_state_dict(checkpoint['policy_net'])
            self.agent.eval()

            # Load world model for planning
            base_path = model_path.replace('_policy.pth', '')
            world_model_path = f"{base_path}_world_model.pth"

            if use_planning and os.path.exists(world_model_path):
                print(f"Loading world model for planning: {world_model_path}")
                self.world_model = WorldModelNetwork(state_dim=95, action_dim=4)
                wm_checkpoint = torch.load(world_model_path, map_location='cpu', weights_only=False)
                self.world_model.load_state_dict(wm_checkpoint['model'])
                self.world_model.eval()
                print(f"  Planning ENABLED: {planning_freq*100:.0f}% frequency, horizon {planning_horizon}")
            elif use_planning:
                print(f"  Warning: World model not found at {world_model_path}")
                print(f"  Planning DISABLED - will use policy only")
                self.use_planning = False

            # Initialize faith population
            # Note: Patterns are created automatically by FaithPopulation
            self.faith_population = FaithPopulation(population_size=20)

            if 'faith_population' in checkpoint:
                print("  Loading faith population from checkpoint...")
                # Update fitness from checkpoint if available
                for i, pattern_data in enumerate(checkpoint['faith_population'][:20]):
                    if i < len(self.faith_population.patterns):
                        self.faith_population.patterns[i].fitness = pattern_data.get('fitness', 0)
                print(f"  Faith population loaded: {len(self.faith_population.patterns)} patterns")
            else:
                print(f"  Created default faith population: 20 patterns")

            # Initialize entity discovery world model
            self.entity_world_model = EntityDiscoveryWorldModel(
                obs_dim=95,
                action_dim=4,
                max_entity_types=20
            )
            if 'entity_world_model' in checkpoint:
                self.entity_world_model.load_state_dict(checkpoint['entity_world_model'])
                print(f"  Entity discovery model loaded")

            print(f"Loaded faith-based evolutionary agent: {model_path}")
            print(f"  Episodes trained: {len(checkpoint.get('episode_rewards', []))}")
            print(f"  Input: 95-dim (92 temporal + 3 context)")
            print(f"  Faith frequency: {faith_freq*100:.0f}%")
            print(f"  Faith discoveries: {checkpoint.get('faith_discovery_count', 0)}")
            print(f"  Entity types discovered: {checkpoint.get('entity_types_discovered', 0)}")
            print(f"  Patterns detected: {checkpoint.get('patterns_detected', 0)}")
            print(f"  Mechanics confirmed: {checkpoint.get('mechanics_confirmed', 0)}")

            # Store checkpoint info
            self.checkpoint_info = {
                'faith_discoveries': checkpoint.get('faith_discovery_count', 0),
                'entity_types': checkpoint.get('entity_types_discovered', 0),
                'patterns': checkpoint.get('patterns_detected', 0),
                'mechanics': checkpoint.get('mechanics_confirmed', 0),
            }
        else:
            print("No agent loaded - manual control only")
            self.checkpoint_info = {}

        # Initialize pygame
        pygame.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.tiny_font = pygame.font.Font(None, 14)
        self.clock = pygame.time.Clock()

        # Current game
        self.current_game = None
        self.game_name = ""
        self.game_state = None
        self.score = 0
        self.steps = 0
        self.use_ai = True if self.agent else False
        self.last_action = -1
        self.last_action_source = 'reactive'

        # Temporal info display
        self.current_obs = None
        self.reward_direction = (0, 0)
        self.danger_trend = 0
        self.progress_rate = 0

        # Faith system display
        self.recent_discoveries = []
        self.recent_entity_detections = []
        self.current_patterns = []
        self.current_mechanics = []

    def switch_game(self, game_type):
        """Switch to a different game"""
        if game_type == 'snake':
            self.current_game = SnakeGame(size=20)
            self.game_name = "SNAKE"
            self.grid_size = 20
            context_name = 'snake'
        elif game_type == 'pacman':
            self.current_game = PacManGame(size=20)
            self.game_name = "PAC-MAN"
            self.grid_size = 20
            context_name = 'balanced'
        elif game_type == 'dungeon':
            self.current_game = DungeonGame(size=20)
            self.game_name = "DUNGEON"
            self.grid_size = 20
            context_name = 'survival'

        self.game_state = self.current_game.reset()
        self.observer.reset()
        self.score = 0
        self.steps = 0
        self.last_action = -1
        self.last_action_source = 'reactive'
        self._update_temporal_info()

        # Reset faith tracking
        self.recent_discoveries = []
        self.recent_entity_detections = []
        self.current_patterns = []
        self.current_mechanics = []
        self.mechanic_detector = MechanicDetector()
        self.pattern_extractor = UniversalPatternExtractor()

        # Resize window - wider for faith info
        width = self.grid_size * self.cell_size + 300  # Extra space for faith info
        height = self.grid_size * self.cell_size + 80
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f'Faith-Based Agent - {self.game_name}')

    def _update_temporal_info(self):
        """Extract temporal information from observation for display"""
        if self.current_obs is not None and len(self.current_obs) >= 92:
            # Reward direction (indices 46-47)
            self.reward_direction = (self.current_obs[46], self.current_obs[47])

            # Danger trend and progress rate
            if len(self.current_obs) > 90:
                self.danger_trend = self.current_obs[88]
                self.progress_rate = self.current_obs[90]

    def _plan_action(self, state):
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
                rollout_return = self._simulate_rollout(state_tensor, action)
                total_return += rollout_return

            avg_return = total_return / num_rollouts

            if avg_return > best_return:
                best_return = avg_return
                best_action = action

        return best_action

    def _simulate_rollout(self, state, first_action):
        """Simulate one trajectory using world model"""
        current_state = state.clone()
        total_return = 0.0
        discount = 1.0
        gamma = 0.99

        with torch.no_grad():
            # Take first action
            action_tensor = torch.LongTensor([first_action])
            next_state, reward, done = self.world_model(current_state, action_tensor)
            total_return += reward.item() * discount
            discount *= gamma

            if done.item() > 0.5:
                return total_return

            current_state = next_state

            # Simulate remaining horizon steps using policy
            for _ in range(self.planning_horizon - 1):
                # Use policy to select action
                q_values = self.agent.get_combined_q(current_state)
                action = q_values.argmax(dim=1).item()

                # Simulate with world model
                action_tensor = torch.LongTensor([action])
                next_state, reward, done = self.world_model(current_state, action_tensor)

                total_return += reward.item() * discount
                discount *= gamma

                if done.item() > 0.5:
                    break

                current_state = next_state

        return total_return

    def _get_faith_action(self):
        """Select action from faith population"""
        if self.faith_population is None or len(self.faith_population.patterns) == 0:
            return np.random.randint(4)

        # Select best performing pattern
        best_pattern = max(self.faith_population.patterns, key=lambda p: p.fitness)

        # Get dominant behavior type from behavior_types dict
        dominant_behavior = max(best_pattern.behavior_types.items(), key=lambda x: x[1])[0]

        # Execute pattern behavior
        if dominant_behavior == 'wait':
            # Wait/no-op - return last action or random
            return self.last_action if self.last_action != -1 else np.random.randint(4)
        elif dominant_behavior == 'explore':
            # Random exploration
            return np.random.randint(4)
        elif dominant_behavior == 'rhythmic':
            # Alternating pattern
            return (self.steps % 4)
        elif dominant_behavior == 'sacrificial':
            # Move towards danger (opposite of safe direction)
            if abs(self.danger_trend) > 0.1:
                # Move in direction of danger
                return np.random.randint(4)
            return np.random.randint(4)
        else:
            return np.random.randint(4)

    def draw_snake(self):
        """Draw snake game"""
        game = self.current_game
        self.screen.fill(BLACK)

        # Draw boundary
        for i in range(self.grid_size):
            pygame.draw.rect(self.screen, DARK_GREEN,
                           (i * self.cell_size, 0, self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, DARK_GREEN,
                           (i * self.cell_size, (self.grid_size-1) * self.cell_size,
                            self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, DARK_GREEN,
                           (0, i * self.cell_size, self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, DARK_GREEN,
                           ((self.grid_size-1) * self.cell_size, i * self.cell_size,
                            self.cell_size, self.cell_size))

        # Draw snake body
        for x, y in game.snake[1:]:
            pygame.draw.rect(self.screen, GREEN,
                           (x * self.cell_size + 2, y * self.cell_size + 2,
                            self.cell_size - 4, self.cell_size - 4))

        # Draw snake head (different color if faith action)
        hx, hy = game.snake[0]
        head_color = MAGENTA if self.last_action_source == 'faith' else CYAN
        pygame.draw.rect(self.screen, head_color,
                       (hx * self.cell_size + 1, hy * self.cell_size + 1,
                        self.cell_size - 2, self.cell_size - 2))

        # Draw food pellets
        for fx, fy in game.food_positions:
            pygame.draw.circle(self.screen, RED,
                             (int((fx + 0.5) * self.cell_size),
                              int((fy + 0.5) * self.cell_size)),
                             int(self.cell_size * 0.4))

        # Draw direction arrow and detection rays
        self._draw_direction_arrow(hx, hy)

    def draw_pacman(self):
        """Draw Pac-Man game"""
        game = self.current_game
        self.screen.fill(BLACK)

        # Draw maze walls
        for x, y in game.walls:
            pygame.draw.rect(self.screen, BLUE,
                           (x * self.cell_size, y * self.cell_size,
                            self.cell_size, self.cell_size))

        # Draw pellets
        for px, py in game.pellets:
            pygame.draw.circle(self.screen, WHITE,
                             (int((px + 0.5) * self.cell_size),
                              int((py + 0.5) * self.cell_size)), 3)

        # Draw ghosts
        for ghost in game.ghosts:
            gx, gy = ghost['pos']
            pygame.draw.circle(self.screen, RED,
                             (int((gx + 0.5) * self.cell_size),
                              int((gy + 0.5) * self.cell_size)),
                             int(self.cell_size * 0.4))

        # Draw Pac-Man (different color if faith action)
        px, py = game.pacman_pos
        pacman_color = MAGENTA if self.last_action_source == 'faith' else YELLOW
        pygame.draw.circle(self.screen, pacman_color,
                         (int((px + 0.5) * self.cell_size),
                          int((py + 0.5) * self.cell_size)),
                         int(self.cell_size * 0.4))

        # Draw direction arrow and detection rays
        self._draw_direction_arrow(px, py)

    def draw_dungeon(self):
        """Draw dungeon game"""
        game = self.current_game
        self.screen.fill(BLACK)

        # Draw walls
        for x, y in game.walls:
            pygame.draw.rect(self.screen, GRAY,
                           (x * self.cell_size, y * self.cell_size,
                            self.cell_size, self.cell_size))

        # Draw treasure
        tx, ty = game.treasure
        pygame.draw.circle(self.screen, GOLD,
                         (int((tx + 0.5) * self.cell_size),
                          int((ty + 0.5) * self.cell_size)), 5)

        # Draw enemies
        for enemy in game.enemies:
            mx, my = enemy['pos']
            pygame.draw.circle(self.screen, RED,
                             (int((mx + 0.5) * self.cell_size),
                              int((my + 0.5) * self.cell_size)),
                             int(self.cell_size * 0.35))

        # Draw player (different color if faith action)
        px, py = game.player_pos
        player_color = MAGENTA if self.last_action_source == 'faith' else GREEN
        pygame.draw.circle(self.screen, player_color,
                         (int((px + 0.5) * self.cell_size),
                          int((py + 0.5) * self.cell_size)),
                         int(self.cell_size * 0.4))

        # Draw direction arrow and detection rays
        self._draw_direction_arrow(px, py)

    def _draw_direction_arrow(self, agent_x, agent_y):
        """Draw arrow showing direction to nearest reward"""
        if abs(self.reward_direction[0]) < 0.01 and abs(self.reward_direction[1]) < 0.01:
            return

        # Agent center position
        center_x = int((agent_x + 0.5) * self.cell_size)
        center_y = int((agent_y + 0.5) * self.cell_size)

        # Arrow direction (scaled)
        arrow_length = self.cell_size * 2
        end_x = center_x + int(self.reward_direction[0] * arrow_length)
        end_y = center_y + int(self.reward_direction[1] * arrow_length)

        # Draw arrow line
        pygame.draw.line(self.screen, PURPLE, (center_x, center_y), (end_x, end_y), 3)

        # Draw arrowhead
        pygame.draw.circle(self.screen, PURPLE, (end_x, end_y), 5)

        # Draw entity and wall detection rays
        self._draw_detection_rays(center_x, center_y)

    def _draw_detection_rays(self, center_x, center_y):
        """Draw rays showing entity and wall detection in 8 directions"""
        if self.current_obs is None or len(self.current_obs) < 48:
            return

        # Ray directions: N, NE, E, SE, S, SW, W, NW
        ray_dirs = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]

        max_ray_length = self.cell_size * 4

        for i, (dx, dy) in enumerate(ray_dirs):
            # Extract distances from state
            base_idx = i * 3
            reward_dist = self.current_obs[base_idx]
            entity_dist = self.current_obs[base_idx + 1]
            wall_dist = self.current_obs[base_idx + 2]

            # Normalize direction
            length = np.sqrt(dx*dx + dy*dy)
            norm_dx = dx / length
            norm_dy = dy / length

            # Draw WALL ray (gray, thin)
            wall_ray_len = wall_dist * max_ray_length
            wall_end_x = center_x + int(norm_dx * wall_ray_len)
            wall_end_y = center_y + int(norm_dy * wall_ray_len)
            pygame.draw.line(self.screen, GRAY, (center_x, center_y),
                           (wall_end_x, wall_end_y), 1)

            # Draw ENTITY ray (orange/red based on distance)
            if entity_dist < 0.9:
                entity_ray_len = entity_dist * max_ray_length
                if entity_dist < 0.3:
                    entity_color = RED
                elif entity_dist < 0.6:
                    entity_color = ORANGE
                else:
                    entity_color = (255, 200, 100)

                entity_end_x = center_x + int(norm_dx * entity_ray_len)
                entity_end_y = center_y + int(norm_dy * entity_ray_len)
                pygame.draw.line(self.screen, entity_color, (center_x, center_y),
                               (entity_end_x, entity_end_y), 2)
                pygame.draw.circle(self.screen, entity_color, (entity_end_x, entity_end_y), 3)

    def draw_info_panel(self):
        """Draw info panel on the right with faith system information"""
        panel_x = self.grid_size * self.cell_size + 10
        y = 10

        # Game name
        title = self.font.render(self.game_name, True, WHITE)
        self.screen.blit(title, (panel_x, y))
        y += 30

        # Mode
        mode_text = 'AI' if self.use_ai else 'MANUAL'
        mode_color = GREEN if self.use_ai else YELLOW
        mode = self.font.render(f'Mode: {mode_text}', True, mode_color)
        self.screen.blit(mode, (panel_x, y))
        y += 25

        # Stats
        stats = [
            f'Score: {self.score}',
            f'Steps: {self.steps}',
        ]

        if hasattr(self.current_game, 'lives'):
            stats.append(f'Lives: {self.current_game.lives}')

        if self.game_name == 'SNAKE':
            stats.append(f'Length: {len(self.current_game.snake)}')
        elif self.game_name == 'PAC-MAN':
            stats.append(f'Pellets: {len(self.current_game.pellets)}')
        elif self.game_name == 'DUNGEON':
            treasure_collected = 1 if self.score >= 10 else 0
            stats.append(f'Treasure: {treasure_collected}/1')

        for stat in stats:
            text = self.small_font.render(stat, True, WHITE)
            self.screen.blit(text, (panel_x, y))
            y += 18

        # Action distribution
        y += 5
        action_title = self.font.render('ACTIONS:', True, CYAN)
        self.screen.blit(action_title, (panel_x, y))
        y += 22

        total_actions = sum(self.action_counts.values()) or 1
        faith_pct = (self.action_counts['faith'] / total_actions) * 100
        planning_pct = (self.action_counts['planning'] / total_actions) * 100
        reactive_pct = (self.action_counts['reactive'] / total_actions) * 100

        # Color current action source
        faith_color = MAGENTA if self.last_action_source == 'faith' else GRAY
        planning_color = CYAN if self.last_action_source == 'planning' else GRAY
        reactive_color = WHITE if self.last_action_source == 'reactive' else GRAY

        faith_text = self.small_font.render(f'Faith: {faith_pct:.1f}%', True, faith_color)
        self.screen.blit(faith_text, (panel_x, y))
        y += 18

        planning_text = self.small_font.render(f'Plan: {planning_pct:.1f}%', True, planning_color)
        self.screen.blit(planning_text, (panel_x, y))
        y += 18

        reactive_text = self.small_font.render(f'React: {reactive_pct:.1f}%', True, reactive_color)
        self.screen.blit(reactive_text, (panel_x, y))
        y += 22

        # Faith system info
        faith_header = self.font.render('FAITH SYSTEM:', True, MAGENTA)
        self.screen.blit(faith_header, (panel_x, y))
        y += 22

        # Training stats from checkpoint
        if self.checkpoint_info:
            disco_text = self.tiny_font.render(
                f"Trained: {self.checkpoint_info['faith_discoveries']} discoveries",
                True, GRAY
            )
            self.screen.blit(disco_text, (panel_x, y))
            y += 16

            entity_text = self.tiny_font.render(
                f"{self.checkpoint_info['entity_types']} entities, " +
                f"{self.checkpoint_info['patterns']} patterns",
                True, GRAY
            )
            self.screen.blit(entity_text, (panel_x, y))
            y += 18

        # Recent discoveries
        if len(self.recent_discoveries) > 0:
            disco_count = self.small_font.render(
                f'Discoveries: {len(self.recent_discoveries)}',
                True, GOLD
            )
            self.screen.blit(disco_count, (panel_x, y))
            y += 18

        # Entity detections
        if len(self.recent_entity_detections) > 0:
            entity_count = self.small_font.render(
                f'Entities: {len(self.recent_entity_detections)}',
                True, TEAL
            )
            self.screen.blit(entity_count, (panel_x, y))
            y += 18

        # Patterns detected
        if len(self.current_patterns) > 0:
            pattern_text = self.small_font.render(
                f'Patterns: {len(self.current_patterns)}',
                True, ORANGE
            )
            self.screen.blit(pattern_text, (panel_x, y))
            y += 18

        # Mechanics confirmed
        if len(self.current_mechanics) > 0:
            mechanic_text = self.small_font.render(
                f'Mechanics: {len(self.current_mechanics)}',
                True, GREEN
            )
            self.screen.blit(mechanic_text, (panel_x, y))
            y += 18

        # Controls at bottom
        y = self.grid_size * self.cell_size + 10
        controls = self.small_font.render(
            '1:Snake 2:PacMan 3:Dungeon', True, GRAY)
        self.screen.blit(controls, (10, y))
        y += 18
        controls2 = self.small_font.render(
            'SPACE:AI R:Reset ESC:Quit', True, GRAY)
        self.screen.blit(controls2, (10, y))
        y += 18
        controls3 = self.tiny_font.render(
            'MAGENTA=Faith Action', True, MAGENTA)
        self.screen.blit(controls3, (10, y))

    def draw(self):
        """Draw current game"""
        if self.game_name == 'SNAKE':
            self.draw_snake()
        elif self.game_name == 'PAC-MAN':
            self.draw_pacman()
        elif self.game_name == 'DUNGEON':
            self.draw_dungeon()

        self.draw_info_panel()
        pygame.display.flip()

    def run(self, speed=10):
        """Main game loop"""
        # Start with snake
        self.switch_game('snake')

        running = True
        while running:
            # Handle events
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.use_ai = not self.use_ai
                        print(f"Mode: {'AI' if self.use_ai else 'Manual'}")
                    elif event.key == pygame.K_r:
                        self.game_state = self.current_game.reset()
                        self.observer.reset()
                        self.score = 0
                        self.steps = 0
                        self._update_temporal_info()
                        print("Reset!")
                    elif event.key == pygame.K_1:
                        self.switch_game('snake')
                        print("Switched to SNAKE")
                    elif event.key == pygame.K_2:
                        self.switch_game('pacman')
                        print("Switched to PAC-MAN")
                    elif event.key == pygame.K_3:
                        self.switch_game('dungeon')
                        print("Switched to DUNGEON")
                    # Manual controls
                    elif event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_DOWN:
                        action = 1
                    elif event.key == pygame.K_LEFT:
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        action = 3

            # AI action with faith-based decision making
            if self.use_ai and action is None and self.agent:
                # Convert game_state to observation
                obs = self.observer.observe(self.game_state)
                self.current_obs = obs

                # Infer context from observation
                self.current_context = infer_context_from_observation(obs)

                # Determine context name
                if self.current_context[0] == 1.0:
                    self.current_context_name = "SNAKE"
                elif self.current_context[1] == 1.0:
                    self.current_context_name = "BALANCED"
                else:
                    self.current_context_name = "SURVIVAL"

                # Add context to observation
                obs_with_context = add_context_to_observation(obs, self.current_context)

                # Decide action source: faith, planning, or reactive
                rand = np.random.random()

                if rand < self.faith_freq:
                    # Faith action
                    action = self._get_faith_action()
                    self.last_action_source = 'faith'
                    self.action_counts['faith'] += 1
                elif self.use_planning and self.world_model and rand < (self.faith_freq + self.planning_freq):
                    # Planning action
                    action = self._plan_action(obs_with_context)
                    self.last_action_source = 'planning'
                    self.action_counts['planning'] += 1
                else:
                    # Reactive action
                    action = self.agent.get_action(obs_with_context, epsilon=0.0)
                    self.last_action_source = 'reactive'
                    self.action_counts['reactive'] += 1

                # Update temporal info for display
                self._update_temporal_info()

                # Update faith system components (passive observation)
                if self.steps % 10 == 0:  # Update every 10 steps
                    # Observe for pattern extraction (needs entity list, using empty for now)
                    self.pattern_extractor.observe([], obs)

                    # Extract patterns
                    patterns = self.pattern_extractor.extract_patterns()
                    if patterns:
                        self.current_patterns = list(patterns.keys())

                    # Test mechanic hypotheses
                    mechanics = self.mechanic_detector.test_hypotheses()
                    if mechanics:
                        self.current_mechanics = [m for m, h in mechanics.items() if h.confidence > 0.7]

            # Take step
            if action is not None:
                prev_score = self.score
                self.game_state, reward, done = self.current_game.step(action)
                self.score = self.game_state.get('score', 0)
                self.steps = self.current_game.steps
                self.last_action = action

                # Track discoveries
                if reward > 0 and self.last_action_source == 'faith':
                    self.recent_discoveries.append({
                        'step': self.steps,
                        'reward': reward
                    })

                # Observe for mechanic detection
                if reward != 0:
                    event = {'type': 'reward', 'value': reward}
                    self.mechanic_detector.observe(reward, action, event)

                if done:
                    print(f"{self.game_name} finished! Score: {self.score} | Context: {self.current_context_name}")
                    print(f"  Faith: {self.action_counts['faith']} | " +
                          f"Planning: {self.action_counts['planning']} | " +
                          f"Reactive: {self.action_counts['reactive']}")
                    print(f"  Discoveries: {len(self.recent_discoveries)} | " +
                          f"Patterns: {len(self.current_patterns)}")

                    # Auto reset
                    self.game_state = self.current_game.reset()
                    self.observer.reset()
                    self.score = 0
                    self.steps = 0

            # Draw
            self.draw()
            self.clock.tick(speed)

        pygame.quit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Faith-Based Evolutionary Visual Test Games')
    parser.add_argument('--model', type=str, default=None, help='Path to faith-based agent model')
    parser.add_argument('--speed', type=int, default=10, help='Game speed (FPS)')
    parser.add_argument('--no-planning', action='store_true', help='Disable world model planning')
    parser.add_argument('--planning-freq', type=float, default=0.3, help='Planning frequency (0-1)')
    parser.add_argument('--planning-horizon', type=int, default=5, help='Planning horizon (steps)')
    parser.add_argument('--faith-freq', type=float, default=0.1, help='Faith action frequency (0-1)')
    args = parser.parse_args()

    print("=" * 70)
    print("FAITH-BASED EVOLUTIONARY VISUAL GAME TESTER")
    print("Test Faith-Based Agent on Snake, Pac-Man, Dungeon")
    print("=" * 70)
    print()
    print("REVOLUTIONARY CAPABILITIES:")
    print("  Faith Patterns - Persistent exploration despite negative feedback")
    print("  Entity Discovery - Learns what entities are without labels")
    print("  Universal Patterns - Game-agnostic strategy transfer")
    print("  Mechanic Detection - Discovers hidden game rules")
    print()
    print("CONTROLS:")
    print("  1 - Switch to Snake")
    print("  2 - Switch to Pac-Man")
    print("  3 - Switch to Dungeon")
    print("  SPACE - Toggle AI/Manual mode")
    print("  R - Reset current game")
    print("  Arrow Keys - Manual control")
    print("  ESC - Quit")
    print()
    print("VISUAL INDICATORS:")
    print("  MAGENTA agent = Faith action currently executing")
    print("  Purple arrow = Direction to nearest reward")
    print("  Gray rays = Wall detection")
    print("  Orange/Red rays = Entity detection (red=close)")
    print()

    if not args.no_planning:
        print("ACTION DISTRIBUTION:")
        print(f"  Faith: {args.faith_freq*100:.0f}% of actions")
        print(f"  Planning: {args.planning_freq*100:.0f}% of actions")
        print(f"  Reactive: {(1.0-args.faith_freq-args.planning_freq)*100:.0f}% of actions")
        print(f"  Planning Horizon: {args.planning_horizon} steps lookahead")
        print()

    runner = FaithVisualRunner(
        model_path=args.model,
        use_planning=not args.no_planning,
        planning_freq=args.planning_freq,
        planning_horizon=args.planning_horizon,
        faith_freq=args.faith_freq
    )
    runner.run(speed=args.speed)


if __name__ == '__main__':
    main()

"""
Multi-Game Switcher - Switch between 4 games with one agent!

Press 1-4 to switch games:
  1 - Snake (Generalization Test)
  2 - Dungeon (Zero-Shot Transfer)
  3 - Local View (Zero-Shot Transfer)
  4 - PacMan (Zero-Shot Transfer)

Other controls:
  UP/DOWN - Change difficulty level
  SPACE - Pause/Resume
  R - Reset episode
  ESC - Quit
"""
import pygame
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.enhanced_snake_game import EnhancedSnakeGame
from src.core.simple_dungeon_game import SimpleDungeonGame
from src.core.local_view_game import LocalViewGame
from src.core.simple_pacman_game import SimplePacManGame
from src.context_aware_agent import ContextAwareDQN, add_context_to_observation
from src.core.expanded_temporal_observer import ExpandedTemporalObserver

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
GOLD = (255, 215, 0)
DARK_GRAY = (64, 64, 64)
DARK_GREEN = (0, 100, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)
LIGHT_GRAY = (180, 180, 180)


class MultiGameSwitcher:
    def __init__(self, model_path, cell_size=25):
        self.cell_size = cell_size
        self.model_path = model_path

        # Load agent once
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.agent = ContextAwareDQN(obs_dim=183, action_dim=4)
        self.agent.load_state_dict(checkpoint['policy_net'])
        self.agent.eval()

        # Observer
        self.observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)

        # Initialize pygame
        pygame.init()
        self.screen_width = 20 * cell_size + 400
        self.screen_height = 20 * cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.font = pygame.font.SysFont('arial', 16)
        self.title_font = pygame.font.SysFont('arial', 20, bold=True)
        self.small_font = pygame.font.SysFont('arial', 12)
        self.clock = pygame.time.Clock()

        # Game selection
        self.current_game_idx = 0
        self.game_names = ["Snake", "Dungeon", "Local View", "PacMan"]
        self.difficulty_level = 3

        # Stats
        self.episodes = 0
        self.victories = 0
        self.deaths = 0

        # Create initial game
        self.create_game()

        print("=" * 80)
        print("MULTI-GAME SWITCHER - One Agent, Four Games!")
        print("=" * 80)
        print(f"Loaded model: {model_path}")
        print()
        print("CONTROLS:")
        print("  1 - Switch to Snake")
        print("  2 - Switch to Dungeon")
        print("  3 - Switch to Local View")
        print("  4 - Switch to PacMan")
        print("  UP/DOWN - Increase/Decrease difficulty")
        print("  SPACE - Pause/Resume")
        print("  R - Reset episode")
        print("  ESC - Quit")
        print("=" * 80)

    def create_game(self):
        """Create the selected game"""
        game_idx = self.current_game_idx
        level = self.difficulty_level

        if game_idx == 0:  # Snake
            self.game = EnhancedSnakeGame(
                size=20,
                initial_pellets=7,
                max_pellets=12,
                food_timeout=150,
                obstacle_level=level,
                max_steps=400,
                max_total_food=40
            )
            pygame.display.set_caption(f"Snake - Obstacle Level {level}")
        elif game_idx == 1:  # Dungeon
            self.game = SimpleDungeonGame(
                size=20,
                num_treasures=3,
                enemy_level=level,
                max_steps=500
            )
            pygame.display.set_caption(f"Dungeon - Enemy Level {level}")
        elif game_idx == 2:  # Local View
            self.game = LocalViewGame(
                world_size=40,
                num_coins=20,
                enemy_level=level,
                max_steps=800,
                viewport_size=25
            )
            pygame.display.set_caption(f"Local View - Enemy Level {level}")
        elif game_idx == 3:  # PacMan
            self.game = SimplePacManGame(
                size=20,
                num_pellets=30,
                ghost_level=level,
                max_steps=500
            )
            pygame.display.set_caption(f"PacMan - Ghost Level {level}")

        self.reset_episode()
        print(f"\nSwitched to: {self.game_names[game_idx]} (Level {level})")

    def switch_game(self, game_idx):
        """Switch to a different game"""
        if 0 <= game_idx < len(self.game_names):
            self.current_game_idx = game_idx
            self.create_game()

    def change_difficulty(self, delta):
        """Change difficulty level"""
        new_level = max(0, self.difficulty_level + delta)
        if new_level != self.difficulty_level:
            self.difficulty_level = new_level
            self.create_game()

    def reset_episode(self):
        """Reset current game"""
        self.state = self.game.reset()
        self.observer.reset()
        self.done = False

    def draw_snake(self):
        """Draw Snake game"""
        # Boundaries
        for i in range(20):
            pygame.draw.rect(self.screen, DARK_GREEN,
                           (i * self.cell_size, 0, self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, DARK_GREEN,
                           (i * self.cell_size, 19 * self.cell_size, self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, DARK_GREEN,
                           (0, i * self.cell_size, self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, DARK_GREEN,
                           (19 * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))

        # Obstacles
        for ox, oy in self.game.central_obstacles:
            pygame.draw.rect(self.screen, YELLOW,
                           (ox * self.cell_size, oy * self.cell_size,
                            self.cell_size, self.cell_size))

        # Snake body
        for x, y in self.game.snake[1:]:
            pygame.draw.rect(self.screen, GREEN,
                           (x * self.cell_size + 2, y * self.cell_size + 2,
                            self.cell_size - 4, self.cell_size - 4))

        # Snake head
        hx, hy = self.game.snake[0]
        pygame.draw.rect(self.screen, CYAN,
                       (hx * self.cell_size + 1, hy * self.cell_size + 1,
                        self.cell_size - 2, self.cell_size - 2))

        # Food
        for fx, fy in self.game.food_positions:
            pygame.draw.circle(self.screen, RED,
                             (int((fx + 0.5) * self.cell_size),
                              int((fy + 0.5) * self.cell_size)),
                             int(self.cell_size * 0.4))

    def draw_dungeon(self):
        """Draw Dungeon game"""
        # Walls
        for wx, wy in self.game.walls:
            pygame.draw.rect(self.screen, DARK_GRAY,
                           (wx * self.cell_size, wy * self.cell_size,
                            self.cell_size, self.cell_size))

        # Treasures
        for tx, ty in self.game.treasures:
            pygame.draw.circle(self.screen, YELLOW,
                             (int((tx + 0.5) * self.cell_size),
                              int((ty + 0.5) * self.cell_size)), 8)
            pygame.draw.circle(self.screen, GOLD,
                             (int((tx + 0.5) * self.cell_size),
                              int((ty + 0.5) * self.cell_size)), 6)

        # Enemies
        enemy_colors = [RED, PURPLE, ORANGE]
        for i, enemy in enumerate(self.game.enemies):
            ex, ey = enemy['pos']
            color = enemy_colors[i % len(enemy_colors)]
            pygame.draw.rect(self.screen, color,
                           (ex * self.cell_size + 2, ey * self.cell_size + 2,
                            self.cell_size - 4, self.cell_size - 4))

        # Player
        px, py = self.game.player_pos
        pygame.draw.circle(self.screen, GREEN,
                         (int((px + 0.5) * self.cell_size),
                          int((py + 0.5) * self.cell_size)),
                         int(self.cell_size * 0.4))

    def draw_local_view(self):
        """Draw Local View game - simplified without minimap"""
        viewport_size = 25
        view_x, view_y = self.get_viewport_bounds()

        # Background grid
        for x in range(min(viewport_size, 20)):
            for y in range(min(viewport_size, 20)):
                pygame.draw.rect(self.screen, DARK_GRAY,
                               (x * self.cell_size, y * self.cell_size,
                                self.cell_size, self.cell_size), 1)

        # Walls
        for wx, wy in self.game.walls:
            if self.is_in_viewport((wx, wy), view_x, view_y):
                sx, sy = self.world_to_screen((wx, wy), view_x, view_y)
                if 0 <= sx < 20 * self.cell_size and 0 <= sy < 20 * self.cell_size:
                    pygame.draw.rect(self.screen, BLUE,
                                   (sx, sy, self.cell_size, self.cell_size))

        # Coins
        for cx, cy in self.game.coins:
            if self.is_in_viewport((cx, cy), view_x, view_y):
                sx, sy = self.world_to_screen((cx, cy), view_x, view_y)
                if 0 <= sx < 20 * self.cell_size and 0 <= sy < 20 * self.cell_size:
                    pygame.draw.circle(self.screen, YELLOW,
                                     (int(sx + self.cell_size/2),
                                      int(sy + self.cell_size/2)),
                                     self.cell_size // 3)

        # Enemies
        for enemy in self.game.enemies:
            ex, ey = enemy['pos']
            if self.is_in_viewport((ex, ey), view_x, view_y):
                sx, sy = self.world_to_screen((ex, ey), view_x, view_y)
                if 0 <= sx < 20 * self.cell_size and 0 <= sy < 20 * self.cell_size:
                    pygame.draw.rect(self.screen, RED,
                                   (sx + 2, sy + 2, self.cell_size - 4, self.cell_size - 4))

        # Agent
        ax, ay = self.game.agent_pos
        sx, sy = self.world_to_screen((ax, ay), view_x, view_y)
        if 0 <= sx < 20 * self.cell_size and 0 <= sy < 20 * self.cell_size:
            pygame.draw.circle(self.screen, GREEN,
                             (int(sx + self.cell_size/2),
                              int(sy + self.cell_size/2)),
                             self.cell_size // 2)

    def get_viewport_bounds(self):
        """Get viewport bounds for local view"""
        ax, ay = self.game.agent_pos
        half_view = 12  # Show 25x25 but fit in 20x20 screen
        return ax - half_view, ay - half_view

    def world_to_screen(self, world_pos, view_x, view_y):
        """Convert world to screen coords"""
        wx, wy = world_pos
        return (wx - view_x) * self.cell_size, (wy - view_y) * self.cell_size

    def is_in_viewport(self, world_pos, view_x, view_y):
        """Check if position is in viewport"""
        wx, wy = world_pos
        return (view_x <= wx < view_x + 20 and view_y <= wy < view_y + 20)

    def draw_pacman(self):
        """Draw PacMan game"""
        # Walls
        for wx, wy in self.game.walls:
            pygame.draw.rect(self.screen, BLUE,
                           (wx * self.cell_size, wy * self.cell_size,
                            self.cell_size, self.cell_size))

        # Pellets
        for px, py in self.game.pellets:
            pygame.draw.circle(self.screen, WHITE,
                             (int((px + 0.5) * self.cell_size),
                              int((py + 0.5) * self.cell_size)), 3)

        # Ghosts
        ghost_colors = [RED, PINK, ORANGE]
        for i, ghost in enumerate(self.game.ghosts):
            gx, gy = ghost['pos']
            color = ghost_colors[i % len(ghost_colors)]
            pygame.draw.circle(self.screen, color,
                             (int((gx + 0.5) * self.cell_size),
                              int((gy + 0.5) * self.cell_size)),
                             int(self.cell_size * 0.4))

        # PacMan
        pmx, pmy = self.game.pacman_pos
        pygame.draw.circle(self.screen, YELLOW,
                         (int((pmx + 0.5) * self.cell_size),
                          int((pmy + 0.5) * self.cell_size)),
                         int(self.cell_size * 0.45))

    def draw_info_panel(self):
        """Draw info panel"""
        panel_x = 20 * self.cell_size + 20
        y = 10

        # Title
        title = self.title_font.render("MULTI-GAME DEMO", True, CYAN)
        self.screen.blit(title, (panel_x, y))
        y += 35

        # Game selector
        selector_text = self.font.render("SELECT GAME:", True, YELLOW)
        self.screen.blit(selector_text, (panel_x, y))
        y += 25

        for i, name in enumerate(self.game_names):
            color = GREEN if i == self.current_game_idx else GRAY
            text = self.font.render(f"{i+1}. {name}", True, color)
            self.screen.blit(text, (panel_x, y))
            y += 22

        y += 15
        level_text = self.font.render(f"Difficulty: {self.difficulty_level}", True, WHITE)
        self.screen.blit(level_text, (panel_x, y))
        y += 30

        # Current episode stats
        episode_title = self.font.render("CURRENT EPISODE:", True, CYAN)
        self.screen.blit(episode_title, (panel_x, y))
        y += 25

        # Game-specific stats
        if self.current_game_idx == 0:  # Snake
            stats = [
                f"Score: {self.game.total_collected}/40",
                f"Food: {len(self.game.food_positions)}",
                f"Length: {len(self.game.snake)}",
                f"Steps: {self.game.steps}",
            ]
        elif self.current_game_idx == 1:  # Dungeon
            stats = [
                f"Treasures: {self.game.score}/3",
                f"Remaining: {len(self.game.treasures)}",
                f"Lives: {self.game.lives}",
                f"Steps: {self.game.steps}",
            ]
        elif self.current_game_idx == 2:  # Local View
            stats = [
                f"Coins: {self.game.score}/20",
                f"Remaining: {len(self.game.coins)}",
                f"Lives: {self.game.lives}",
                f"Steps: {self.game.steps}",
            ]
        elif self.current_game_idx == 3:  # PacMan
            stats = [
                f"Pellets: {self.game.score}/30",
                f"Remaining: {len(self.game.pellets)}",
                f"Lives: {self.game.lives}",
                f"Steps: {self.game.steps}",
            ]

        for stat in stats:
            text = self.font.render(stat, True, WHITE)
            self.screen.blit(text, (panel_x, y))
            y += 20

        # Overall stats
        y += 15
        overall_title = self.font.render("OVERALL STATS:", True, GREEN)
        self.screen.blit(overall_title, (panel_x, y))
        y += 22

        overall = [
            f"Episodes: {self.episodes}",
            f"Victories: {self.victories}",
            f"Deaths: {self.deaths}",
        ]

        if self.episodes > 0:
            win_rate = (self.victories / self.episodes) * 100
            overall.append(f"Win rate: {win_rate:.1f}%")

        for stat in overall:
            text = self.font.render(stat, True, WHITE)
            self.screen.blit(text, (panel_x, y))
            y += 20

    def draw_game(self):
        """Draw current game"""
        self.screen.fill(BLACK)

        if self.current_game_idx == 0:
            self.draw_snake()
        elif self.current_game_idx == 1:
            self.draw_dungeon()
        elif self.current_game_idx == 2:
            self.draw_local_view()
        elif self.current_game_idx == 3:
            self.draw_pacman()

        self.draw_info_panel()
        pygame.display.flip()

    def run(self, speed=15):
        """Main loop"""
        running = True
        paused = False

        while running:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("PAUSED" if paused else "RESUMED")
                    elif event.key == pygame.K_r:
                        self.reset_episode()
                        print("RESET!")
                    elif event.key == pygame.K_UP:
                        self.change_difficulty(1)
                    elif event.key == pygame.K_DOWN:
                        self.change_difficulty(-1)
                    elif event.key == pygame.K_1:
                        self.switch_game(0)
                    elif event.key == pygame.K_2:
                        self.switch_game(1)
                    elif event.key == pygame.K_3:
                        self.switch_game(2)
                    elif event.key == pygame.K_4:
                        self.switch_game(3)

            # AI step (if not paused)
            if not paused and not self.done:
                # Get observation
                obs = self.observer.observe(self.state)

                # Use balanced context
                context = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
                obs_with_context = add_context_to_observation(obs, context)

                # Get action
                action = self.agent.get_action(obs_with_context, epsilon=0.0)

                # Step game
                self.state, reward, self.done = self.game.step(action)

                if self.done:
                    self.episodes += 1

                    # Check victory condition based on game
                    victory = False
                    if self.current_game_idx == 0:  # Snake
                        victory = self.game.total_collected >= 40
                    elif self.current_game_idx == 1:  # Dungeon
                        victory = len(self.game.treasures) == 0
                    elif self.current_game_idx == 2:  # Local View
                        victory = len(self.game.coins) == 0
                    elif self.current_game_idx == 3:  # PacMan
                        victory = len(self.game.pellets) == 0

                    if victory:
                        self.victories += 1
                        print(f"Episode {self.episodes}: VICTORY in {self.game_names[self.current_game_idx]}!")
                    else:
                        self.deaths += 1
                        print(f"Episode {self.episodes}: Failed in {self.game_names[self.current_game_idx]}")

                    # Auto-reset after 2 seconds
                    pygame.time.wait(2000)
                    self.reset_episode()

            # Draw
            self.draw_game()
            self.clock.tick(speed)

        # Final stats
        print("\n" + "=" * 80)
        print("MULTI-GAME SESSION COMPLETE")
        print("=" * 80)
        print(f"Episodes: {self.episodes}")
        print(f"Victories: {self.victories}")
        print(f"Deaths: {self.deaths}")
        if self.episodes > 0:
            print(f"Win Rate: {(self.victories/self.episodes)*100:.1f}%")
        print("=" * 80)

        pygame.quit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--speed', type=int, default=15, help='Game speed (FPS)')
    args = parser.parse_args()

    demo = MultiGameSwitcher(args.model)
    demo.run(speed=args.speed)

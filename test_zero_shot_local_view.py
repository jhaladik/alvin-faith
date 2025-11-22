"""
Zero-Shot Transfer Test: Snake Agent → Local View Collector

Test if snake agent can play with moving perspective (camera follows agent)
"""
import pygame
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.local_view_game import LocalViewGame
from src.context_aware_agent import ContextAwareDQN, add_context_to_observation
from src.core.expanded_temporal_observer import ExpandedTemporalObserver

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
CYAN = (0, 255, 255)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
GOLD = (255, 215, 0)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (180, 180, 180)
ORANGE = (255, 165, 0)
PURPLE = (200, 0, 255)


class ZeroShotLocalViewTest:
    def __init__(self, model_path, enemy_level=0, cell_size=20):
        self.cell_size = cell_size
        self.viewport_size = 25  # Show 25x25 grid on screen
        self.enemy_level = enemy_level

        # Load SNAKE agent (no Local View training!)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.agent = ContextAwareDQN(obs_dim=183, action_dim=4)
        self.agent.load_state_dict(checkpoint['policy_net'])
        self.agent.eval()

        # Create Local View game (40x40 world)
        self.game = LocalViewGame(
            world_size=40,
            num_coins=20,
            enemy_level=enemy_level,
            max_steps=800,
            viewport_size=self.viewport_size
        )

        # Observer (same as snake training!)
        self.observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)

        # Initialize pygame
        main_width = self.viewport_size * cell_size
        main_height = self.viewport_size * cell_size
        panel_width = 350
        minimap_size = 200

        self.screen = pygame.display.set_mode((main_width + panel_width, main_height))
        pygame.display.set_caption(f"Zero-Shot: Snake→LocalView (Level {enemy_level})")
        self.font = pygame.font.SysFont('arial', 16)
        self.title_font = pygame.font.SysFont('arial', 20, bold=True)
        self.small_font = pygame.font.SysFont('arial', 12)
        self.clock = pygame.time.Clock()

        # Minimap settings
        self.minimap_size = minimap_size
        self.minimap_scale = minimap_size / self.game.world_size

        # Stats
        self.episodes_completed = 0
        self.total_victories = 0
        self.total_deaths = 0
        self.best_score = 0

        # Human play mode
        self.human_playing = False
        self.current_direction = 0  # Default: UP

        # Reset game
        self.state = self.game.reset()
        self.observer.reset()
        self.done = False

        print("=" * 80)
        print("ZERO-SHOT TRANSFER TEST: Snake Agent -> Local View Collector")
        print("=" * 80)
        print(f"Loaded SNAKE checkpoint: {model_path}")
        print(f"Testing on: Local View Game with {len(self.game.enemies)} enemies (level {enemy_level})")
        print(f"World size: {self.game.world_size}x{self.game.world_size}")
        print(f"Viewport: {self.viewport_size}x{self.viewport_size} (moving camera)")
        print()
        print("KEY QUESTION: Can agent handle moving perspective?")
        print("  Snake learned: Avoid walls, collect food (global view)")
        print("  This needs: Navigate large world, limited visibility, moving camera")
        print()
        print("CONTROLS:")
        print("  M - Toggle AI/Human mode")
        print("  ARROW KEYS - Steer (in human mode)")
        print("  UP/DOWN - Increase/Decrease difficulty (in AI mode)")
        print("  R - Reset episode")
        print("  SPACE - Pause/Resume")
        print("  ESC - Quit")
        print("=" * 80)

    def get_viewport_bounds(self):
        """Calculate what portion of world to render"""
        ax, ay = self.game.agent_pos
        half_view = self.viewport_size // 2

        # Center viewport on agent
        view_x = ax - half_view
        view_y = ay - half_view

        return view_x, view_y

    def world_to_screen(self, world_pos):
        """Convert world coordinates to screen coordinates"""
        wx, wy = world_pos
        view_x, view_y = self.get_viewport_bounds()

        screen_x = (wx - view_x) * self.cell_size
        screen_y = (wy - view_y) * self.cell_size

        return screen_x, screen_y

    def is_in_viewport(self, world_pos):
        """Check if world position is visible in viewport"""
        view_x, view_y = self.get_viewport_bounds()
        wx, wy = world_pos

        return (view_x <= wx < view_x + self.viewport_size and
                view_y <= wy < view_y + self.viewport_size)

    def draw_game(self):
        """Draw the game with moving perspective"""
        self.screen.fill(BLACK)

        main_width = self.viewport_size * self.cell_size

        # Draw background grid
        for x in range(self.viewport_size):
            for y in range(self.viewport_size):
                pygame.draw.rect(self.screen, DARK_GRAY,
                               (x * self.cell_size, y * self.cell_size,
                                self.cell_size, self.cell_size), 1)

        # Get viewport bounds
        view_x, view_y = self.get_viewport_bounds()

        # Draw walls (only those in viewport)
        for wx, wy in self.game.walls:
            if self.is_in_viewport((wx, wy)):
                sx, sy = self.world_to_screen((wx, wy))
                pygame.draw.rect(self.screen, BLUE,
                               (sx, sy, self.cell_size, self.cell_size))

        # Draw coins (only those in viewport)
        for cx, cy in self.game.coins:
            if self.is_in_viewport((cx, cy)):
                sx, sy = self.world_to_screen((cx, cy))
                # Coin with glow
                pygame.draw.circle(self.screen, YELLOW,
                                 (int(sx + self.cell_size/2),
                                  int(sy + self.cell_size/2)),
                                 self.cell_size // 3)
                pygame.draw.circle(self.screen, GOLD,
                                 (int(sx + self.cell_size/2),
                                  int(sy + self.cell_size/2)),
                                 self.cell_size // 4)

        # Draw enemies (only those in viewport)
        enemy_colors = [RED, ORANGE, PURPLE]
        for i, enemy in enumerate(self.game.enemies):
            ex, ey = enemy['pos']
            if self.is_in_viewport((ex, ey)):
                sx, sy = self.world_to_screen((ex, ey))
                color = enemy_colors[i % len(enemy_colors)]

                # Enemy body
                pygame.draw.rect(self.screen, color,
                               (sx + 2, sy + 2,
                                self.cell_size - 4, self.cell_size - 4))
                # Eyes
                pygame.draw.circle(self.screen, WHITE,
                                 (int(sx + self.cell_size * 0.35),
                                  int(sy + self.cell_size * 0.4)), 2)
                pygame.draw.circle(self.screen, WHITE,
                                 (int(sx + self.cell_size * 0.65),
                                  int(sy + self.cell_size * 0.4)), 2)

        # Draw agent (always at center of viewport)
        ax, ay = self.game.agent_pos
        sx, sy = self.world_to_screen((ax, ay))
        # Agent with glow (highlight)
        pygame.draw.circle(self.screen, CYAN,
                         (int(sx + self.cell_size/2),
                          int(sy + self.cell_size/2)),
                         self.cell_size // 2 + 2, 2)
        pygame.draw.circle(self.screen, GREEN,
                         (int(sx + self.cell_size/2),
                          int(sy + self.cell_size/2)),
                         self.cell_size // 2 - 1)

        # Draw minimap
        self.draw_minimap(main_width + 20, 10)

        # Draw info panel
        self.draw_info_panel(main_width + 20, 220)

        pygame.display.flip()

    def draw_minimap(self, x, y):
        """Draw minimap showing full world"""
        # Background
        pygame.draw.rect(self.screen, DARK_GRAY,
                       (x, y, self.minimap_size, self.minimap_size))

        # Draw viewport bounds on minimap
        ax, ay = self.game.agent_pos
        half_view = self.viewport_size // 2
        view_x = ax - half_view
        view_y = ay - half_view
        viewport_rect = (
            int(view_x * self.minimap_scale),
            int(view_y * self.minimap_scale),
            int(self.viewport_size * self.minimap_scale),
            int(self.viewport_size * self.minimap_scale)
        )
        pygame.draw.rect(self.screen, YELLOW,
                       (x + viewport_rect[0], y + viewport_rect[1],
                        viewport_rect[2], viewport_rect[3]), 2)

        # Draw coins on minimap
        for cx, cy in self.game.coins:
            mini_x = int(x + cx * self.minimap_scale)
            mini_y = int(y + cy * self.minimap_scale)
            pygame.draw.circle(self.screen, GOLD, (mini_x, mini_y), 2)

        # Draw enemies on minimap
        for enemy in self.game.enemies:
            ex, ey = enemy['pos']
            mini_x = int(x + ex * self.minimap_scale)
            mini_y = int(y + ey * self.minimap_scale)
            pygame.draw.circle(self.screen, RED, (mini_x, mini_y), 3)

        # Draw agent on minimap
        mini_ax = int(x + ax * self.minimap_scale)
        mini_ay = int(y + ay * self.minimap_scale)
        pygame.draw.circle(self.screen, GREEN, (mini_ax, mini_ay), 4)

        # Minimap label
        label = self.small_font.render("MINIMAP", True, WHITE)
        self.screen.blit(label, (x, y - 15))

    def draw_info_panel(self, x, y):
        """Draw information panel"""
        # Title
        title = self.title_font.render("ZERO-SHOT TRANSFER", True, CYAN)
        self.screen.blit(title, (x, y))
        y += 30

        # Mode info
        mode_color = ORANGE if self.human_playing else GREEN
        mode_text = "HUMAN PLAYING" if self.human_playing else "AI PLAYING"
        mode_render = self.font.render(f"Mode: {mode_text}", True, mode_color)
        self.screen.blit(mode_render, (x, y))
        y += 22

        # Agent info
        agent_text = self.font.render("Agent: SNAKE", True, GRAY)
        self.screen.blit(agent_text, (x, y))
        y += 20

        game_text = self.font.render(f"Game: Local View L{self.enemy_level}", True, WHITE)
        self.screen.blit(game_text, (x, y))
        y += 25

        # Current episode
        episode_title = self.font.render("CURRENT EPISODE:", True, YELLOW)
        self.screen.blit(episode_title, (x, y))
        y += 22

        current_stats = [
            f"Coins: {self.game.score}/{self.game.num_coins}",
            f"Remaining: {len(self.game.coins)}",
            f"Lives: {self.game.lives}",
            f"Steps: {self.game.steps}/{self.game.max_steps}",
            f"Position: ({self.game.agent_pos[0]}, {self.game.agent_pos[1]})",
        ]

        for stat in current_stats:
            text = self.font.render(stat, True, WHITE)
            self.screen.blit(text, (x, y))
            y += 20

        # Overall stats
        y += 10
        overall_title = self.font.render("OVERALL STATS:", True, GREEN)
        self.screen.blit(overall_title, (x, y))
        y += 22

        overall_stats = [
            f"Episodes: {self.episodes_completed}",
            f"Victories: {self.total_victories}",
            f"Deaths: {self.total_deaths}",
            f"Best: {self.best_score}/{self.game.num_coins}",
        ]

        if self.episodes_completed > 0:
            win_rate = (self.total_victories / self.episodes_completed) * 100
            overall_stats.append(f"Win rate: {win_rate:.1f}%")

        for stat in overall_stats:
            text = self.font.render(stat, True, WHITE)
            self.screen.blit(text, (x, y))
            y += 20

        # Enemy info
        y += 10
        if self.game.enemies:
            enemy_text = self.font.render(f"Enemies: {len(self.game.enemies)}", True, RED)
            self.screen.blit(enemy_text, (x, y))
            y += 20
            behavior = self.game.enemies[0]['behavior']
            behavior_text = self.small_font.render(f"AI: {behavior}", True, GRAY)
            self.screen.blit(behavior_text, (x, y))

    def reset_episode(self):
        """Reset for new episode"""
        self.state = self.game.reset()
        self.observer.reset()
        self.done = False

    def change_enemy_level(self, delta):
        """Change enemy difficulty"""
        new_level = max(0, min(6, self.enemy_level + delta))
        if new_level != self.enemy_level:
            self.enemy_level = new_level
            self.game = LocalViewGame(
                world_size=40,
                num_coins=20,
                enemy_level=new_level,
                max_steps=800
            )
            self.reset_episode()
            behavior = self.game.enemies[0]['behavior'] if self.game.enemies else 'none'
            print(f"\nEnemy level: {new_level} ({len(self.game.enemies)} enemies, {behavior})")

    def run(self, speed=10):
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
                    elif event.key == pygame.K_m:
                        # Toggle mode
                        self.human_playing = not self.human_playing
                        mode_str = "HUMAN" if self.human_playing else "AI"
                        print(f"\nMode switched to: {mode_str}")
                    elif self.human_playing:
                        # Human mode - arrow keys change direction
                        if event.key == pygame.K_UP:
                            self.current_direction = 0
                        elif event.key == pygame.K_DOWN:
                            self.current_direction = 1
                        elif event.key == pygame.K_LEFT:
                            self.current_direction = 2
                        elif event.key == pygame.K_RIGHT:
                            self.current_direction = 3
                    else:
                        # AI mode - arrow keys change difficulty
                        if event.key == pygame.K_UP:
                            self.change_enemy_level(1)
                        elif event.key == pygame.K_DOWN:
                            self.change_enemy_level(-1)

            # Game step (if not paused)
            if not paused and not self.done:
                # Get observation (using SNAKE observer!)
                obs = self.observer.observe(self.state)

                # Get action based on mode
                if self.human_playing:
                    # Human mode - continuous movement
                    action = self.current_direction
                else:
                    # AI mode
                    context = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
                    obs_with_context = add_context_to_observation(obs, context)
                    action = self.agent.get_action(obs_with_context, epsilon=0.0)

                # Step game
                self.state, reward, self.done = self.game.step(action)

                if self.done:
                    self.episodes_completed += 1
                    self.best_score = max(self.best_score, self.game.score)

                    if self.game.score >= self.game.num_coins or len(self.game.coins) == 0:
                        self.total_victories += 1
                        print(f"Episode {self.episodes_completed}: VICTORY! ({self.game.score}/{self.game.num_coins})")
                    else:
                        self.total_deaths += 1
                        print(f"Episode {self.episodes_completed}: Failed (Score: {self.game.score}/{self.game.num_coins})")

                    # Auto-reset after 2 seconds
                    pygame.time.wait(2000)
                    self.reset_episode()

            # Draw
            self.draw_game()
            self.clock.tick(speed)

        # Final stats
        print("\n" + "=" * 80)
        print("ZERO-SHOT TRANSFER RESULTS")
        print("=" * 80)
        print(f"Enemy Level: {self.enemy_level} ({len(self.game.enemies)} enemies)")
        print(f"Episodes: {self.episodes_completed}")
        print(f"Victories: {self.total_victories}")
        print(f"Deaths: {self.total_deaths}")
        print(f"Best Score: {self.best_score}/{self.game.num_coins}")
        if self.episodes_completed > 0:
            print(f"Win Rate: {(self.total_victories/self.episodes_completed)*100:.1f}%")
        print()
        print("CONCLUSION:")
        if self.episodes_completed > 0 and self.total_victories > 0:
            print("  Success! Snake agent handles moving perspective!")
            print("  Spatial reasoning transfers despite viewport changes!")
        else:
            print("  Transfer unsuccessful - may need viewport-specific training")
        print("=" * 80)

        pygame.quit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to SNAKE model checkpoint')
    parser.add_argument('--enemy-level', type=int, default=0, help='Enemy difficulty (0-4)')
    parser.add_argument('--speed', type=int, default=10, help='Game speed (FPS)')
    args = parser.parse_args()

    pygame.init()
    demo = ZeroShotLocalViewTest(args.model, enemy_level=args.enemy_level, cell_size=20)
    demo.run(speed=args.speed)

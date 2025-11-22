"""
Zero-Shot Transfer Test: Snake Agent -> Dungeon

Test if snake agent can play Dungeon without ANY training!

Progressive difficulty:
- Level 0: No enemies (just collect treasures)
- Level 1: 1 patrolling enemy
- Level 2: 2 patrolling enemies
- Level 3: 3 patrolling enemies
- Level 4+: Smarter enemy patrol patterns
"""
import pygame
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.simple_dungeon_game import SimpleDungeonGame
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
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)

class ZeroShotDungeonTest:
    def __init__(self, model_path, enemy_level=0, cell_size=25):
        self.cell_size = cell_size
        self.grid_size = 20
        self.enemy_level = enemy_level

        # Load SNAKE agent (no Dungeon training!)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.agent = ContextAwareDQN(obs_dim=183, action_dim=4)
        self.agent.load_state_dict(checkpoint['policy_net'])
        self.agent.eval()

        # Create Dungeon game
        self.game = SimpleDungeonGame(
            size=20,
            num_treasures=3,
            enemy_level=enemy_level,
            max_steps=500
        )

        # Observer (same as snake training!)
        self.observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)

        # Initialize pygame
        pygame.init()
        width = self.grid_size * cell_size + 400
        height = self.grid_size * cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Zero-Shot Transfer: Snake->Dungeon (Level {enemy_level})")
        self.font = pygame.font.SysFont('arial', 18)
        self.title_font = pygame.font.SysFont('arial', 22, bold=True)
        self.clock = pygame.time.Clock()

        # Stats
        self.episodes_completed = 0
        self.total_victories = 0
        self.total_deaths = 0
        self.best_score = 0

        # Reset game
        self.state = self.game.reset()
        self.observer.reset()
        self.done = False

        print("=" * 70)
        print("ZERO-SHOT TRANSFER TEST: Snake Agent -> Dungeon")
        print("=" * 70)
        print(f"Loaded SNAKE checkpoint: {model_path}")
        print(f"Testing on: Dungeon with {len(self.game.enemies)} enemies (level {enemy_level})")
        print()
        print("KEY QUESTION: Can spatial reasoning transfer to dungeon exploration?")
        print("  Snake learned: Avoid walls, collect food")
        print("  Dungeon needs: Navigate maze, collect treasure, avoid enemies")
        print()
        print("CONTROLS:")
        print("  UP/DOWN - Increase/Decrease enemy difficulty")
        print("  R - Reset episode")
        print("  SPACE - Pause/Resume")
        print("  ESC - Quit")
        print("=" * 70)

    def draw_game(self):
        """Draw the game state"""
        self.screen.fill(BLACK)

        # Draw walls (dungeon maze)
        for wx, wy in self.game.walls:
            pygame.draw.rect(self.screen, DARK_GRAY,
                           (wx * self.cell_size, wy * self.cell_size,
                            self.cell_size, self.cell_size))
            # Add border for depth
            pygame.draw.rect(self.screen, GRAY,
                           (wx * self.cell_size, wy * self.cell_size,
                            self.cell_size, self.cell_size), 1)

        # Draw treasures (gold coins!)
        for tx, ty in self.game.treasures:
            # Outer glow
            pygame.draw.circle(self.screen, YELLOW,
                             (int((tx + 0.5) * self.cell_size),
                              int((ty + 0.5) * self.cell_size)), 8)
            # Inner gold
            pygame.draw.circle(self.screen, GOLD,
                             (int((tx + 0.5) * self.cell_size),
                              int((ty + 0.5) * self.cell_size)), 6)

        # Draw enemies (patrolling monsters)
        enemy_colors = [RED, PURPLE, ORANGE]
        for i, enemy in enumerate(self.game.enemies):
            ex, ey = enemy['pos']
            color = enemy_colors[i % len(enemy_colors)]
            # Body
            pygame.draw.rect(self.screen, color,
                           (int(ex * self.cell_size + 2),
                            int(ey * self.cell_size + 2),
                            self.cell_size - 4, self.cell_size - 4))
            # Eyes
            pygame.draw.circle(self.screen, WHITE,
                             (int((ex + 0.3) * self.cell_size),
                              int((ey + 0.4) * self.cell_size)), 3)
            pygame.draw.circle(self.screen, WHITE,
                             (int((ex + 0.7) * self.cell_size),
                              int((ey + 0.4) * self.cell_size)), 3)

        # Draw player (hero/adventurer)
        px, py = self.game.player_pos
        # Body (green adventurer)
        pygame.draw.circle(self.screen, GREEN,
                         (int((px + 0.5) * self.cell_size),
                          int((py + 0.5) * self.cell_size)),
                         int(self.cell_size * 0.4))
        # Highlight
        pygame.draw.circle(self.screen, CYAN,
                         (int((px + 0.5) * self.cell_size),
                          int((py + 0.5) * self.cell_size)),
                         int(self.cell_size * 0.4), 2)

        # Draw info panel
        panel_x = self.grid_size * self.cell_size + 20
        y = 10

        # Title
        title = self.title_font.render("ZERO-SHOT TRANSFER", True, CYAN)
        self.screen.blit(title, (panel_x, y))
        y += 30

        # Agent info
        agent_text = self.font.render("Agent: SNAKE (no Dungeon training!)", True, GRAY)
        self.screen.blit(agent_text, (panel_x, y))
        y += 25

        # Game info
        game_text = self.font.render(f"Game: Dungeon Level {self.enemy_level}", True, WHITE)
        self.screen.blit(game_text, (panel_x, y))
        y += 20

        enemy_count = self.font.render(f"Enemies: {len(self.game.enemies)}", True, RED)
        self.screen.blit(enemy_count, (panel_x, y))
        y += 35

        # Current episode
        episode_title = self.font.render("CURRENT EPISODE:", True, GOLD)
        self.screen.blit(episode_title, (panel_x, y))
        y += 25

        current_stats = [
            f"Treasures: {self.game.score}/{self.game.num_treasures}",
            f"Remaining: {len(self.game.treasures)}",
            f"Lives: {self.game.lives}",
            f"Steps: {self.game.steps}/{self.game.max_steps}",
        ]

        for stat in current_stats:
            text = self.font.render(stat, True, WHITE)
            self.screen.blit(text, (panel_x, y))
            y += 22

        # Overall stats
        y += 10
        overall_title = self.font.render("TRANSFER PERFORMANCE:", True, GREEN)
        self.screen.blit(overall_title, (panel_x, y))
        y += 25

        overall_stats = [
            f"Episodes: {self.episodes_completed}",
            f"Victories: {self.total_victories}",
            f"Deaths: {self.total_deaths}",
            f"Best score: {self.best_score}/{self.game.num_treasures}",
        ]

        if self.episodes_completed > 0:
            win_rate = (self.total_victories / self.episodes_completed) * 100
            overall_stats.append(f"Win rate: {win_rate:.1f}%")

        for stat in overall_stats:
            text = self.font.render(stat, True, WHITE)
            self.screen.blit(text, (panel_x, y))
            y += 22

        # Enemy behavior info
        y += 10
        if self.game.enemies:
            behavior = self.game.enemies[0]['behavior']
            behavior_text = self.font.render(f"Enemy AI: {behavior}", True, RED)
            self.screen.blit(behavior_text, (panel_x, y))

        pygame.display.flip()

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
            self.game = SimpleDungeonGame(
                size=20,
                num_treasures=3,
                enemy_level=new_level,
                max_steps=500
            )
            self.reset_episode()
            behavior = self.game.enemies[0]['behavior'] if self.game.enemies else 'none'
            print(f"\nEnemy level changed to {new_level} ({len(self.game.enemies)} enemies, {behavior})")

    def run(self, speed=8):
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
                        self.change_enemy_level(1)
                    elif event.key == pygame.K_DOWN:
                        self.change_enemy_level(-1)

            # AI step (if not paused)
            if not paused and not self.done:
                # Get observation (using SNAKE observer!)
                obs = self.observer.observe(self.state)

                # Use 'balanced' context (not 'snake' - more general)
                context = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
                obs_with_context = add_context_to_observation(obs, context)

                # Get action from SNAKE agent
                action = self.agent.get_action(obs_with_context, epsilon=0.0)

                # Step game
                self.state, reward, self.done = self.game.step(action)

                if self.done:
                    self.episodes_completed += 1
                    self.best_score = max(self.best_score, self.game.score)

                    if self.game.score >= self.game.num_treasures or len(self.game.treasures) == 0:
                        self.total_victories += 1
                        print(f"Episode {self.episodes_completed}: VICTORY! ({self.game.score}/{self.game.num_treasures})")
                    else:
                        self.total_deaths += 1
                        print(f"Episode {self.episodes_completed}: Died (Score: {self.game.score}/{self.game.num_treasures})")

                    # Auto-reset after 2 seconds
                    pygame.time.wait(2000)
                    self.reset_episode()

            # Draw
            self.draw_game()
            self.clock.tick(speed)

        # Final stats
        print("\n" + "=" * 70)
        print("ZERO-SHOT TRANSFER RESULTS")
        print("=" * 70)
        print(f"Enemy Level: {self.enemy_level} ({len(self.game.enemies)} enemies)")
        print(f"Episodes: {self.episodes_completed}")
        print(f"Victories: {self.total_victories}")
        print(f"Deaths: {self.total_deaths}")
        print(f"Best Score: {self.best_score}/{self.game.num_treasures}")
        if self.episodes_completed > 0:
            print(f"Win Rate: {(self.total_victories/self.episodes_completed)*100:.1f}%")
        print()
        print("CONCLUSION:")
        if self.episodes_completed > 0 and self.total_victories > 0:
            print("  ✓ Snake agent CAN explore dungeons without training!")
            print("  ✓ Spatial reasoning transfers to maze navigation!")
        else:
            print("  ✗ Transfer unsuccessful - needs game-specific training")
        print("=" * 70)

        pygame.quit()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to SNAKE model checkpoint')
    parser.add_argument('--enemy-level', type=int, default=0, help='Enemy difficulty (0=none, 1-3=patrol, 4+=smart)')
    parser.add_argument('--speed', type=int, default=8, help='Game speed (FPS)')
    args = parser.parse_args()

    demo = ZeroShotDungeonTest(args.model, enemy_level=args.enemy_level)
    demo.run(speed=args.speed)

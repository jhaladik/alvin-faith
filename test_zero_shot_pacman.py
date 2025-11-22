"""
Zero-Shot Transfer Test: Snake Agent → PacMan

Test if snake agent can play PacMan without ANY training!

Progressive difficulty:
- Level 0: No ghosts (just collect pellets)
- Level 1: 1 ghost (moving wall)
- Level 2: 2 ghosts
- Level 3: 3 ghosts
- Level 4+: Ghosts chase you
"""
import pygame
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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
PINK = (255, 192, 203)
ORANGE = (255, 165, 0)

class ZeroShotPacManTest:
    def __init__(self, model_path, ghost_level=0, cell_size=25):
        self.cell_size = cell_size
        self.grid_size = 20
        self.ghost_level = ghost_level

        # Load SNAKE agent (no PacMan training!)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.agent = ContextAwareDQN(obs_dim=183, action_dim=4)
        self.agent.load_state_dict(checkpoint['policy_net'])
        self.agent.eval()

        # Create PacMan game
        self.game = SimplePacManGame(
            size=20,
            num_pellets=30,
            ghost_level=ghost_level,
            max_steps=500
        )

        # Observer (same as snake training!)
        self.observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)

        # Initialize pygame
        pygame.init()
        width = self.grid_size * cell_size + 400
        height = self.grid_size * cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Zero-Shot Transfer: Snake→PacMan (Level {ghost_level})")
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
        print("ZERO-SHOT TRANSFER TEST: Snake Agent → PacMan")
        print("=" * 70)
        print(f"Loaded SNAKE checkpoint: {model_path}")
        print(f"Testing on: PacMan with {len(self.game.ghosts)} ghosts (level {ghost_level})")
        print()
        print("KEY QUESTION: Can spatial reasoning transfer?")
        print("  Snake learned: Avoid walls, collect food")
        print("  PacMan needs: Avoid walls, avoid ghosts, collect pellets")
        print()
        print("CONTROLS:")
        print("  UP/DOWN - Increase/Decrease ghost difficulty")
        print("  R - Reset episode")
        print("  SPACE - Pause/Resume")
        print("  ESC - Quit")
        print("=" * 70)

    def draw_game(self):
        """Draw the game state"""
        self.screen.fill(BLACK)

        # Draw walls (maze)
        for wx, wy in self.game.walls:
            pygame.draw.rect(self.screen, BLUE,
                           (wx * self.cell_size, wy * self.cell_size,
                            self.cell_size, self.cell_size))

        # Draw pellets
        for px, py in self.game.pellets:
            pygame.draw.circle(self.screen, WHITE,
                             (int((px + 0.5) * self.cell_size),
                              int((py + 0.5) * self.cell_size)), 3)

        # Draw ghosts (different colors for each)
        ghost_colors = [RED, PINK, ORANGE]
        for i, ghost in enumerate(self.game.ghosts):
            gx, gy = ghost['pos']
            color = ghost_colors[i % len(ghost_colors)]
            pygame.draw.circle(self.screen, color,
                             (int((gx + 0.5) * self.cell_size),
                              int((gy + 0.5) * self.cell_size)),
                             int(self.cell_size * 0.4))
            # Draw eyes
            pygame.draw.circle(self.screen, WHITE,
                             (int((gx + 0.3) * self.cell_size),
                              int((gy + 0.4) * self.cell_size)), 2)
            pygame.draw.circle(self.screen, WHITE,
                             (int((gx + 0.7) * self.cell_size),
                              int((gy + 0.4) * self.cell_size)), 2)

        # Draw PacMan
        pmx, pmy = self.game.pacman_pos
        pygame.draw.circle(self.screen, YELLOW,
                         (int((pmx + 0.5) * self.cell_size),
                          int((pmy + 0.5) * self.cell_size)),
                         int(self.cell_size * 0.45))

        # Draw info panel
        panel_x = self.grid_size * self.cell_size + 20
        y = 10

        # Title
        title = self.title_font.render("ZERO-SHOT TRANSFER", True, CYAN)
        self.screen.blit(title, (panel_x, y))
        y += 30

        # Agent info
        agent_text = self.font.render("Agent: SNAKE (no PacMan training!)", True, GRAY)
        self.screen.blit(agent_text, (panel_x, y))
        y += 25

        # Game info
        game_text = self.font.render(f"Game: PacMan Level {self.ghost_level}", True, WHITE)
        self.screen.blit(game_text, (panel_x, y))
        y += 20

        ghost_count = self.font.render(f"Ghosts: {len(self.game.ghosts)}", True, RED)
        self.screen.blit(ghost_count, (panel_x, y))
        y += 35

        # Current episode
        episode_title = self.font.render("CURRENT EPISODE:", True, YELLOW)
        self.screen.blit(episode_title, (panel_x, y))
        y += 25

        current_stats = [
            f"Pellets: {self.game.score}/{self.game.num_pellets}",
            f"Remaining: {len(self.game.pellets)}",
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
            f"Best score: {self.best_score}/{self.game.num_pellets}",
        ]

        if self.episodes_completed > 0:
            win_rate = (self.total_victories / self.episodes_completed) * 100
            overall_stats.append(f"Win rate: {win_rate:.1f}%")

        for stat in overall_stats:
            text = self.font.render(stat, True, WHITE)
            self.screen.blit(text, (panel_x, y))
            y += 22

        # Ghost behavior info
        y += 10
        if self.game.ghosts:
            behavior = self.game.ghosts[0]['behavior']
            behavior_text = self.font.render(f"Ghost AI: {behavior}", True, RED)
            self.screen.blit(behavior_text, (panel_x, y))

        pygame.display.flip()

    def reset_episode(self):
        """Reset for new episode"""
        self.state = self.game.reset()
        self.observer.reset()
        self.done = False

    def change_ghost_level(self, delta):
        """Change ghost difficulty"""
        new_level = max(0, min(6, self.ghost_level + delta))
        if new_level != self.ghost_level:
            self.ghost_level = new_level
            self.game = SimplePacManGame(
                size=20,
                num_pellets=30,
                ghost_level=new_level,
                max_steps=500
            )
            self.reset_episode()
            behavior = self.game.ghosts[0]['behavior'] if self.game.ghosts else 'none'
            print(f"\nGhost level changed to {new_level} ({len(self.game.ghosts)} ghosts, {behavior})")

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
                        self.change_ghost_level(1)
                    elif event.key == pygame.K_DOWN:
                        self.change_ghost_level(-1)

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

                    if self.game.score >= self.game.num_pellets or len(self.game.pellets) == 0:
                        self.total_victories += 1
                        print(f"Episode {self.episodes_completed}: VICTORY! ({self.game.score}/{self.game.num_pellets})")
                    else:
                        self.total_deaths += 1
                        print(f"Episode {self.episodes_completed}: Died (Score: {self.game.score}/{self.game.num_pellets})")

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
        print(f"Ghost Level: {self.ghost_level} ({len(self.game.ghosts)} ghosts)")
        print(f"Episodes: {self.episodes_completed}")
        print(f"Victories: {self.total_victories}")
        print(f"Deaths: {self.total_deaths}")
        print(f"Best Score: {self.best_score}/{self.game.num_pellets}")
        if self.episodes_completed > 0:
            print(f"Win Rate: {(self.total_victories/self.episodes_completed)*100:.1f}%")
        print()
        print("CONCLUSION:")
        if self.episodes_completed > 0 and self.total_victories > 0:
            print("  ✓ Snake agent CAN play PacMan without training!")
            print("  ✓ Spatial reasoning transfers across games!")
        else:
            print("  ✗ Transfer unsuccessful - needs game-specific training")
        print("=" * 70)

        pygame.quit()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to SNAKE model checkpoint')
    parser.add_argument('--ghost-level', type=int, default=0, help='Ghost difficulty (0=none, 1-3=moving walls, 4+=chase)')
    parser.add_argument('--speed', type=int, default=8, help='Game speed (FPS)')
    args = parser.parse_args()

    demo = ZeroShotPacManTest(args.model, ghost_level=args.ghost_level)
    demo.run(speed=args.speed)

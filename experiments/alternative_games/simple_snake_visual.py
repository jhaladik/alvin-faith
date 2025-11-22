"""
Simple Snake Visual Demo - Clean visualization without overlapping text
"""
import pygame
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.enhanced_snake_game import EnhancedSnakeGame
from src.context_aware_agent import ContextAwareDQN, add_context_to_observation
from src.core.expanded_temporal_observer import ExpandedTemporalObserver

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
GRAY = (128, 128, 128)
DARK_GREEN = (0, 100, 0)

class SimpleSnakeVisual:
    def __init__(self, model_path, cell_size=30):
        self.cell_size = cell_size
        self.grid_size = 20

        # Load agent
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.agent = ContextAwareDQN(obs_dim=183, action_dim=4)
        self.agent.load_state_dict(checkpoint['policy_net'])
        self.agent.eval()

        # Create game
        self.game = EnhancedSnakeGame(
            size=20,
            initial_pellets=7,
            max_pellets=12,
            food_timeout=150,
            obstacle_level=2,
            max_steps=400,
            max_total_food=40
        )

        # Observer
        self.observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)

        # Initialize pygame
        pygame.init()
        width = self.grid_size * cell_size + 300  # Extra space for info
        height = self.grid_size * cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake AI - 40 Pellet Victory Challenge")
        self.font = pygame.font.SysFont('arial', 20)
        self.clock = pygame.time.Clock()

        # Reset game
        self.state = self.game.reset()
        self.observer.reset()
        self.done = False

        print("=" * 60)
        print("SIMPLE SNAKE VISUAL DEMO")
        print("=" * 60)
        print(f"Loaded checkpoint: {model_path}")
        print(f"Episodes trained: {len(checkpoint.get('episode_rewards', []))}")
        print()
        print("CONTROLS:")
        print("  SPACE - Pause/Resume")
        print("  R - Reset")
        print("  ESC - Quit")
        print()
        print("GOAL: Collect 40 pellets to win!")
        print("=" * 60)

    def draw_game(self):
        """Draw the game state"""
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

        # Draw obstacles
        if hasattr(self.game, 'central_obstacles'):
            for ox, oy in self.game.central_obstacles:
                pygame.draw.rect(self.screen, GRAY,
                               (ox * self.cell_size, oy * self.cell_size,
                                self.cell_size, self.cell_size))

        # Draw snake body
        for x, y in self.game.snake[1:]:
            pygame.draw.rect(self.screen, GREEN,
                           (x * self.cell_size + 2, y * self.cell_size + 2,
                            self.cell_size - 4, self.cell_size - 4))

        # Draw snake head
        hx, hy = self.game.snake[0]
        pygame.draw.rect(self.screen, CYAN,
                       (hx * self.cell_size + 1, hy * self.cell_size + 1,
                        self.cell_size - 2, self.cell_size - 2))

        # Draw food
        for fx, fy in self.game.food_positions:
            pygame.draw.circle(self.screen, RED,
                             (int((fx + 0.5) * self.cell_size),
                              int((fy + 0.5) * self.cell_size)),
                             int(self.cell_size * 0.4))

        # Draw info panel
        panel_x = self.grid_size * self.cell_size + 20
        y = 20

        info_lines = [
            f"SCORE: {self.game.total_collected}/40",
            f"",
            f"Food on board: {len(self.game.food_positions)}",
            f"Snake length: {len(self.game.snake)}",
            f"Lives: {self.game.lives}",
            f"Steps: {self.game.steps}",
            f"",
            f"Target: 40 pellets",
            f"Progress: {int(self.game.total_collected/40*100)}%"
        ]

        for line in info_lines:
            if line:  # Skip empty lines
                text = self.font.render(line, True, WHITE)
                self.screen.blit(text, (panel_x, y))
            y += 25

        # Victory message
        if self.game.total_collected >= 40:
            y += 20
            victory_text = self.font.render("VICTORY!", True, GREEN)
            self.screen.blit(victory_text, (panel_x, y))

        pygame.display.flip()

    def run(self, speed=5):
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
                        self.state = self.game.reset()
                        self.observer.reset()
                        self.done = False
                        print("RESET!")

            # AI step (if not paused)
            if not paused and not self.done:
                # Get observation
                obs = self.observer.observe(self.state)
                context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
                obs_with_context = add_context_to_observation(obs, context)

                # Get action
                action = self.agent.get_action(obs_with_context, epsilon=0.0)

                # Step game
                self.state, reward, self.done = self.game.step(action)

                if self.done:
                    print(f"Episode finished! Score: {self.game.total_collected}/40")
                    if self.game.total_collected >= 40:
                        print("VICTORY! Agent reached the goal!")

            # Draw
            self.draw_game()
            self.clock.tick(speed)

        pygame.quit()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--speed', type=int, default=5, help='Game speed (FPS)')
    args = parser.parse_args()

    demo = SimpleSnakeVisual(args.model)
    demo.run(speed=args.speed)

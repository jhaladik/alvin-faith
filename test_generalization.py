"""
Test Agent Generalization - Can it handle MORE obstacles than trained on?

Trained on: obstacle_level=2 (8 obstacles)
Test on: obstacle_level=3, 4, 5 (more obstacles)
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
YELLOW = (255, 255, 0)

class GeneralizationTest:
    def __init__(self, model_path, obstacle_level=3, cell_size=30):
        self.cell_size = cell_size
        self.grid_size = 20
        self.obstacle_level = obstacle_level

        # Load agent
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.agent = ContextAwareDQN(obs_dim=183, action_dim=4)
        self.agent.load_state_dict(checkpoint['policy_net'])
        self.agent.eval()

        # Create game with MORE obstacles than training
        self.game = EnhancedSnakeGame(
            size=20,
            initial_pellets=7,
            max_pellets=12,
            food_timeout=150,
            obstacle_level=obstacle_level,  # MORE than training!
            max_steps=400,
            max_total_food=40
        )

        # Observer
        self.observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)

        # Initialize pygame
        pygame.init()
        width = self.grid_size * cell_size + 350
        height = self.grid_size * cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Generalization Test - Obstacle Level {obstacle_level}")
        self.font = pygame.font.SysFont('arial', 18)
        self.title_font = pygame.font.SysFont('arial', 22, bold=True)
        self.clock = pygame.time.Clock()

        # Stats
        self.episodes_completed = 0
        self.total_victories = 0
        self.total_wall_deaths = 0
        self.total_obstacle_deaths = 0

        # Reset game
        self.state = self.game.reset()
        self.observer.reset()
        self.done = False
        self.death_reason = None

        print("=" * 70)
        print("GENERALIZATION TEST - Can Agent Handle More Obstacles?")
        print("=" * 70)
        print(f"Loaded checkpoint: {model_path}")
        print(f"Trained on: obstacle_level=2 (8 obstacles)")
        print(f"Testing on: obstacle_level={obstacle_level} ({len(self.game.central_obstacles)} obstacles)")
        print()
        print("CONTROLS:")
        print("  UP/DOWN - Increase/Decrease obstacle level")
        print("  R - Reset episode")
        print("  SPACE - Pause/Resume")
        print("  ESC - Quit")
        print()
        print(f"Obstacle positions: {sorted(list(self.game.central_obstacles))[:20]}...")
        print("=" * 70)

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

        # Draw obstacles (highlight them in yellow to see them clearly)
        if hasattr(self.game, 'central_obstacles'):
            for ox, oy in self.game.central_obstacles:
                pygame.draw.rect(self.screen, YELLOW,
                               (ox * self.cell_size, oy * self.cell_size,
                                self.cell_size, self.cell_size))
                # Add border to make them more visible
                pygame.draw.rect(self.screen, WHITE,
                               (ox * self.cell_size, oy * self.cell_size,
                                self.cell_size, self.cell_size), 1)

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
        y = 10

        # Title
        title = self.title_font.render("GENERALIZATION TEST", True, YELLOW)
        self.screen.blit(title, (panel_x, y))
        y += 30

        # Training info
        trained_text = self.font.render("Trained: Level 2 (8 obs)", True, GRAY)
        self.screen.blit(trained_text, (panel_x, y))
        y += 25

        # Current test
        test_text = self.font.render(f"Testing: Level {self.obstacle_level}", True, WHITE)
        self.screen.blit(test_text, (panel_x, y))
        y += 20

        obs_count = self.font.render(f"Obstacles: {len(self.game.central_obstacles)}", True, YELLOW)
        self.screen.blit(obs_count, (panel_x, y))
        y += 35

        # Current episode stats
        episode_title = self.font.render("CURRENT EPISODE:", True, CYAN)
        self.screen.blit(episode_title, (panel_x, y))
        y += 25

        current_stats = [
            f"Score: {self.game.total_collected}/40",
            f"Food: {len(self.game.food_positions)}",
            f"Length: {len(self.game.snake)}",
            f"Steps: {self.game.steps}",
        ]

        for stat in current_stats:
            text = self.font.render(stat, True, WHITE)
            self.screen.blit(text, (panel_x, y))
            y += 22

        # Overall stats
        y += 10
        overall_title = self.font.render("OVERALL STATS:", True, CYAN)
        self.screen.blit(overall_title, (panel_x, y))
        y += 25

        overall_stats = [
            f"Episodes: {self.episodes_completed}",
            f"Victories: {self.total_victories}",
            f"Wall deaths: {self.total_wall_deaths}",
            f"Obstacle deaths: {self.total_obstacle_deaths}",
        ]

        if self.episodes_completed > 0:
            win_rate = (self.total_victories / self.episodes_completed) * 100
            overall_stats.append(f"Win rate: {win_rate:.1f}%")

        for stat in overall_stats:
            text = self.font.render(stat, True, WHITE)
            self.screen.blit(text, (panel_x, y))
            y += 22

        # Death reason
        if self.death_reason:
            y += 10
            reason_text = self.font.render(self.death_reason, True, RED)
            self.screen.blit(reason_text, (panel_x, y))

        pygame.display.flip()

    def check_death_reason(self, prev_state, action):
        """Determine what caused death"""
        head = self.game.snake[0] if self.game.snake else None
        if not head:
            return "Unknown"

        # Check if hit obstacle
        if head in self.game.central_obstacles:
            return "Hit obstacle!"

        # Check if hit wall
        if (head[0] <= 0 or head[0] >= self.game.size-1 or
            head[1] <= 0 or head[1] >= self.game.size-1):
            return "Hit wall!"

        # Check self collision
        if len(self.game.snake) > 1 and head in self.game.snake[1:]:
            return "Self collision!"

        return "Unknown death"

    def reset_episode(self):
        """Reset for new episode"""
        self.state = self.game.reset()
        self.observer.reset()
        self.done = False
        self.death_reason = None

    def change_obstacle_level(self, delta):
        """Change obstacle level"""
        new_level = max(0, self.obstacle_level + delta)
        if new_level != self.obstacle_level:
            self.obstacle_level = new_level
            self.game = EnhancedSnakeGame(
                size=20,
                initial_pellets=7,
                max_pellets=12,
                food_timeout=150,
                obstacle_level=new_level,
                max_steps=400,
                max_total_food=40
            )
            self.reset_episode()
            print(f"\nObstacle level changed to {new_level} ({len(self.game.central_obstacles)} obstacles)")

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
                        self.reset_episode()
                        print("RESET!")
                    elif event.key == pygame.K_UP:
                        self.change_obstacle_level(1)
                    elif event.key == pygame.K_DOWN:
                        self.change_obstacle_level(-1)

            # AI step (if not paused)
            if not paused and not self.done:
                # Get observation
                obs = self.observer.observe(self.state)
                context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
                obs_with_context = add_context_to_observation(obs, context)

                # Get action
                action = self.agent.get_action(obs_with_context, epsilon=0.0)

                # Store previous state for death analysis
                prev_total = self.game.total_collected

                # Step game
                self.state, reward, self.done = self.game.step(action)

                if self.done:
                    self.episodes_completed += 1

                    if self.game.total_collected >= 40:
                        self.total_victories += 1
                        self.death_reason = "VICTORY! (40/40)"
                        print(f"Episode {self.episodes_completed}: VICTORY!")
                    else:
                        # Determine death cause
                        self.death_reason = self.check_death_reason(self.state, action)
                        if "obstacle" in self.death_reason.lower():
                            self.total_obstacle_deaths += 1
                        elif "wall" in self.death_reason.lower():
                            self.total_wall_deaths += 1

                        print(f"Episode {self.episodes_completed}: {self.death_reason} (Score: {self.game.total_collected}/40)")

                    # Auto-reset after 2 seconds
                    pygame.time.wait(2000)
                    self.reset_episode()

            # Draw
            self.draw_game()
            self.clock.tick(speed)

        # Final stats
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        print(f"Obstacle Level: {self.obstacle_level} ({len(self.game.central_obstacles)} obstacles)")
        print(f"Episodes Completed: {self.episodes_completed}")
        print(f"Victories: {self.total_victories}")
        print(f"Wall Deaths: {self.total_wall_deaths}")
        print(f"Obstacle Deaths: {self.total_obstacle_deaths}")
        if self.episodes_completed > 0:
            print(f"Win Rate: {(self.total_victories/self.episodes_completed)*100:.1f}%")
        print("=" * 70)

        pygame.quit()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--obstacle-level', type=int, default=3, help='Obstacle level (trained on 2)')
    parser.add_argument('--speed', type=int, default=5, help='Game speed (FPS)')
    args = parser.parse_args()

    demo = GeneralizationTest(args.model, obstacle_level=args.obstacle_level)
    demo.run(speed=args.speed)

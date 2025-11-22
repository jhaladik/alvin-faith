"""
Gradio Demo with Pygame Graphics
Captures smooth Pygame rendering and streams to Gradio interface
"""

import gradio as gr
import pygame
import torch
import numpy as np
import sys
import os
from PIL import Image
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.enhanced_snake_game import EnhancedSnakeGame
from src.core.simple_pacman_game import SimplePacManGame
from src.core.simple_dungeon_game import SimpleDungeonGame
from src.core.local_view_game import LocalViewGame
from src.context_aware_agent import ContextAwareDQN, add_context_to_observation
from src.core.expanded_temporal_observer import ExpandedTemporalObserver

# Initialize Pygame (headless mode for server)
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Headless mode
pygame.init()

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
DARK_GREEN = (0, 100, 0)
GOLD = (255, 215, 0)


class PygameGameRenderer:
    """Renders games using Pygame for smooth graphics"""

    @staticmethod
    def render_snake(game, cell_size=25):
        """Render Snake with Pygame"""
        size = game.size
        surface = pygame.Surface((size * cell_size, size * cell_size))
        surface.fill(BLACK)

        # Walls - green border
        for i in range(size):
            pygame.draw.rect(surface, DARK_GREEN, (i * cell_size, 0, cell_size, cell_size))
            pygame.draw.rect(surface, DARK_GREEN, (i * cell_size, (size-1) * cell_size, cell_size, cell_size))
            pygame.draw.rect(surface, DARK_GREEN, (0, i * cell_size, cell_size, cell_size))
            pygame.draw.rect(surface, DARK_GREEN, ((size-1) * cell_size, i * cell_size, cell_size, cell_size))

        # Obstacles - gray blocks
        if hasattr(game, 'central_obstacles'):
            for ox, oy in game.central_obstacles:
                pygame.draw.rect(surface, GRAY, (ox * cell_size, oy * cell_size, cell_size, cell_size))

        # Food - red circles
        if hasattr(game, 'food_positions'):
            for fx, fy in game.food_positions:
                pygame.draw.circle(surface, RED,
                                 (int((fx + 0.5) * cell_size), int((fy + 0.5) * cell_size)),
                                 int(cell_size * 0.4))

        # Snake body - green
        if hasattr(game, 'snake') and game.snake:
            for x, y in game.snake[1:]:
                pygame.draw.rect(surface, GREEN,
                               (x * cell_size + 2, y * cell_size + 2,
                                cell_size - 4, cell_size - 4))

            # Snake head - cyan
            if game.snake:
                hx, hy = game.snake[0]
                pygame.draw.rect(surface, CYAN,
                               (hx * cell_size + 1, hy * cell_size + 1,
                                cell_size - 2, cell_size - 2))

        return surface

    @staticmethod
    def render_pacman(game, cell_size=25):
        """Render Pac-Man with Pygame"""
        size = game.size
        surface = pygame.Surface((size * cell_size, size * cell_size))
        surface.fill(BLACK)

        # Walls - blue blocks
        for wx, wy in game.walls:
            pygame.draw.rect(surface, BLUE, (wx * cell_size, wy * cell_size, cell_size, cell_size))

        # Pellets - white dots
        for px, py in game.pellets:
            pygame.draw.circle(surface, WHITE,
                             (int((px + 0.5) * cell_size), int((py + 0.5) * cell_size)), 3)

        # Ghosts - smooth circles with eyes
        ghost_colors = [RED, PINK, ORANGE]
        for i, ghost in enumerate(game.ghosts):
            gx, gy = ghost['pos']
            color = ghost_colors[i % len(ghost_colors)]
            pygame.draw.circle(surface, color,
                             (int((gx + 0.5) * cell_size), int((gy + 0.5) * cell_size)),
                             int(cell_size * 0.4))
            # Eyes
            pygame.draw.circle(surface, WHITE,
                             (int((gx + 0.3) * cell_size), int((gy + 0.4) * cell_size)), 2)
            pygame.draw.circle(surface, WHITE,
                             (int((gx + 0.7) * cell_size), int((gy + 0.4) * cell_size)), 2)

        # Pac-Man - yellow circle
        pmx, pmy = game.pacman_pos
        pygame.draw.circle(surface, YELLOW,
                         (int((pmx + 0.5) * cell_size), int((pmy + 0.5) * cell_size)),
                         int(cell_size * 0.45))

        return surface

    @staticmethod
    def render_dungeon(game, cell_size=25):
        """Render Dungeon with Pygame"""
        size = game.size
        surface = pygame.Surface((size * cell_size, size * cell_size))
        surface.fill(BLACK)

        # Walls - gray blocks
        for wx, wy in game.walls:
            pygame.draw.rect(surface, GRAY, (wx * cell_size, wy * cell_size, cell_size, cell_size))

        # Treasures - gold circles
        for tx, ty in game.treasures:
            pygame.draw.circle(surface, GOLD,
                             (int((tx + 0.5) * cell_size), int((ty + 0.5) * cell_size)),
                             int(cell_size * 0.4))

        # Enemies - red/purple/orange
        enemy_colors = [RED, (160, 32, 240), ORANGE]
        for i, enemy in enumerate(game.enemies):
            ex, ey = enemy['pos']
            color = enemy_colors[i % len(enemy_colors)]
            pygame.draw.rect(surface, color,
                           (ex * cell_size + 2, ey * cell_size + 2,
                            cell_size - 4, cell_size - 4))

        # Player - green circle
        px, py = game.player_pos
        pygame.draw.circle(surface, GREEN,
                         (int((px + 0.5) * cell_size), int((py + 0.5) * cell_size)),
                         int(cell_size * 0.42))

        return surface

    @staticmethod
    def render_local_view(game, cell_size=20, viewport_size=25):
        """Render Sky Collector with Pygame"""
        surface = pygame.Surface((viewport_size * cell_size, viewport_size * cell_size))
        surface.fill((135, 206, 250))  # Sky blue

        ax, ay = game.agent_pos
        half_view = viewport_size // 2
        view_x, view_y = ax - half_view, ay - half_view

        def to_screen(wx, wy):
            return (wx - view_x) * cell_size, (wy - view_y) * cell_size

        def in_view(wx, wy):
            return view_x <= wx < view_x + viewport_size and view_y <= wy < view_y + viewport_size

        # Walls - cyan blocks
        for wx, wy in game.walls:
            if in_view(wx, wy):
                sx, sy = to_screen(wx, wy)
                pygame.draw.rect(surface, CYAN, (sx, sy, cell_size, cell_size))

        # Coins - gold
        for cx_pos, cy_pos in game.coins:
            if in_view(cx_pos, cy_pos):
                sx, sy = to_screen(cx_pos, cy_pos)
                pygame.draw.circle(surface, GOLD,
                                 (int(sx + cell_size/2), int(sy + cell_size/2)), 5)

        # Enemies - red
        for enemy in game.enemies:
            ex, ey = enemy['pos']
            if in_view(ex, ey):
                sx, sy = to_screen(ex, ey)
                pygame.draw.rect(surface, RED,
                               (sx + 2, sy + 2, cell_size - 4, cell_size - 4))

        # Player - green
        sx, sy = to_screen(ax, ay)
        pygame.draw.circle(surface, GREEN,
                         (int(sx + cell_size/2), int(sy + cell_size/2)),
                         int(cell_size * 0.45))

        return surface

    @staticmethod
    def surface_to_image(surface):
        """Convert Pygame surface to PIL Image"""
        # Get pygame surface as bytes (updated method)
        raw_bytes = pygame.image.tobytes(surface, 'RGB')
        # Convert to PIL Image
        image = Image.frombytes('RGB', surface.get_size(), raw_bytes)
        return image


class PygameDemo:
    """Demo manager with Pygame rendering"""

    def __init__(self, model_path='checkpoints/multi_game_enhanced_20251121_190832_policy.pth'):
        self.model_path = model_path
        self.agent = None
        self.observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)
        self.game = None
        self.stats = {'score': 0, 'steps': 0, 'done': False, 'episodes': 0, 'victories': 0}
        self.load_model()

    def load_model(self):
        """Load AI model"""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                self.agent = ContextAwareDQN(obs_dim=183, action_dim=4)
                self.agent.load_state_dict(checkpoint['policy_net'])
                self.agent.eval()
                return "✓ Model Loaded"
            except Exception as e:
                return f"✗ Error: {e}"
        return "⚠ Random Mode"

    def reset_game(self, game_type, difficulty=1):
        """Reset game"""
        self.stats = {'score': 0, 'steps': 0, 'done': False,
                     'episodes': self.stats.get('episodes', 0),
                     'victories': self.stats.get('victories', 0)}

        if game_type == 'snake':
            size = [10, 15, 20][difficulty]
            pellets = [3, 5, 7][difficulty]
            self.game = EnhancedSnakeGame(size=size, initial_pellets=pellets, max_steps=400)
        elif game_type == 'pacman':
            self.game = SimplePacManGame(size=20, num_pellets=30, ghost_level=difficulty, max_steps=500)
        elif game_type == 'dungeon':
            self.game = SimpleDungeonGame(size=20, num_treasures=3, enemy_level=difficulty, max_steps=600)
        elif game_type == 'local_view':
            self.game = LocalViewGame(world_size=40, num_coins=20, enemy_level=difficulty, max_steps=1000)

        self.game.reset()
        self.observer.reset()
        return self.render(game_type), self.get_stats_html(game_type)

    def step(self, game_type):
        """Take one step"""
        if self.stats['done']:
            return self.render(game_type), self.get_stats_html(game_type), "Game Over - Reset to continue"

        # Get action
        state = self._get_state_dict()
        obs = self.observer.observe(state)
        context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        obs_with_context = add_context_to_observation(obs, context)

        if self.agent:
            action = self.agent.get_action(obs_with_context, epsilon=0.0)
        else:
            action = np.random.randint(4)

        # Execute
        _, reward, done = self.game.step(action)
        self.stats['steps'] += 1
        self.stats['score'] = self.game.score
        self.stats['done'] = done

        if done:
            self.stats['episodes'] += 1
            if self.game.score >= getattr(self.game, 'num_pellets', getattr(self.game, 'num_treasures', 999)):
                self.stats['victories'] += 1
                status = f"🎉 Victory! Score: {self.game.score}"
            else:
                status = f"💀 Game Over - Score: {self.game.score}"
        else:
            status = f"Step {self.stats['steps']}"

        return self.render(game_type), self.get_stats_html(game_type), status

    def _get_state_dict(self):
        """Convert game to state dict"""
        state = {
            'grid_size': (getattr(self.game, 'size', getattr(self.game, 'world_size', 20)),) * 2,
            'walls': self.game.walls,
            'score': self.game.score,
            'done': False
        }

        if hasattr(self.game, 'snake'):
            state['agent_pos'] = self.game.snake[0]
            state['snake_body'] = self.game.snake[1:]
            state['rewards'] = list(self.game.food_positions)
        elif hasattr(self.game, 'pacman_pos'):
            state['agent_pos'] = self.game.pacman_pos
            state['rewards'] = list(self.game.pellets)
            state['entities'] = [{'pos': g['pos']} for g in self.game.ghosts]
        elif hasattr(self.game, 'player_pos'):
            state['agent_pos'] = self.game.player_pos
            state['rewards'] = list(self.game.treasures)
            state['entities'] = [{'pos': e['pos']} for e in self.game.enemies]
        elif hasattr(self.game, 'agent_pos'):
            state['agent_pos'] = self.game.agent_pos
            state['rewards'] = list(self.game.coins)
            state['entities'] = [{'pos': e['pos']} for e in self.game.enemies]

        return state

    def render(self, game_type):
        """Render game with Pygame"""
        if game_type == 'snake':
            surface = PygameGameRenderer.render_snake(self.game, cell_size=25)
        elif game_type == 'pacman':
            surface = PygameGameRenderer.render_pacman(self.game, cell_size=25)
        elif game_type == 'dungeon':
            surface = PygameGameRenderer.render_dungeon(self.game, cell_size=25)
        elif game_type == 'local_view':
            surface = PygameGameRenderer.render_local_view(self.game, cell_size=20)

        return PygameGameRenderer.surface_to_image(surface)

    def get_stats_html(self, game_type):
        """Generate stats HTML"""
        game_names = {'snake': '🐍 Snake', 'pacman': '👾 Pac-Man',
                      'dungeon': '🏰 Dungeon', 'local_view': '✈️ Sky Collector'}

        win_rate = (self.stats['victories'] / self.stats['episodes'] * 100) if self.stats['episodes'] > 0 else 0

        html = f"""
        <div style="font-family: 'Courier New', monospace; background: #0a0a0a; color: #fff;
                    padding: 20px; border: 2px solid #333; border-radius: 10px;">
            <h2 style="color: #00ff00; text-align: center; margin: 0 0 20px 0;">
                {game_names.get(game_type, 'Game')}
            </h2>

            <div style="background: #1a1a1a; padding: 15px; margin: 10px 0; border-radius: 8px;">
                <h3 style="color: #ffff00; margin: 0 0 10px 0;">📊 Current Episode</h3>
                <p style="margin: 5px 0;"><span style="color: #888;">Score:</span>
                   <span style="color: #00ff00; font-size: 24px; font-weight: bold;">{self.stats['score']}</span></p>
                <p style="margin: 5px 0;"><span style="color: #888;">Steps:</span>
                   <span style="color: #00ffff;">{self.stats['steps']}</span></p>
                <p style="margin: 5px 0;"><span style="color: #888;">Status:</span>
                   <span style="color: {'#ff0000' if self.stats['done'] else '#00ff00'};">
                   {'DONE' if self.stats['done'] else 'PLAYING'}</span></p>
            </div>

            <div style="background: #1a1a1a; padding: 15px; margin: 10px 0; border-radius: 8px;">
                <h3 style="color: #00ffff; margin: 0 0 10px 0;">🏆 Overall Stats</h3>
                <p style="margin: 5px 0;"><span style="color: #888;">Episodes:</span>
                   <span style="color: #fff;">{self.stats['episodes']}</span></p>
                <p style="margin: 5px 0;"><span style="color: #888;">Victories:</span>
                   <span style="color: #00ff00;">{self.stats['victories']}</span></p>
                <p style="margin: 5px 0;"><span style="color: #888;">Win Rate:</span>
                   <span style="color: #ffff00; font-weight: bold;">{win_rate:.1f}%</span></p>
            </div>

            <div style="background: #1a1a1a; padding: 15px; margin: 10px 0; border-radius: 8px; text-align: center;">
                <h3 style="color: #ff00ff; margin: 0 0 10px 0;">🤖 AI Agent</h3>
                <p style="color: #00ff00; font-size: 16px; font-weight: bold;">
                    {'ACTIVE' if self.agent else 'RANDOM MODE'}
                </p>
                <p style="color: #888; font-size: 12px; margin-top: 5px;">
                    Zero-Shot Transfer Learning
                </p>
            </div>
        </div>
        """
        return html


# Global instance
demo_instance = None


def create_demo():
    """Create Gradio interface with Pygame rendering"""
    global demo_instance

    custom_css = """
    body { background: #000; }
    .gradio-container { background: #0a0a0a !important; }
    """

    with gr.Blocks(title="Pygame Multi-Game AI Demo", css=custom_css) as demo:

        gr.HTML("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(90deg, #1a1a1a, #2a2a2a, #1a1a1a);
                   border-bottom: 3px solid #00ff00;">
            <h1 style="color: #00ff00; font-family: 'Courier New', monospace; margin: 0; font-size: 36px;">
                🎮 MULTI-GAME AI FOUNDATION AGENT
            </h1>
            <p style="color: #00ffff; font-family: 'Courier New', monospace; margin: 10px 0 0 0; font-size: 18px;">
                Zero-Shot Transfer Learning • Smooth Pygame Graphics
            </p>
        </div>
        """)

        with gr.Row():
            # Left: Game display
            with gr.Column(scale=2):
                game_display = gr.Image(label="Game View", type="pil", height=550)
                status_text = gr.Textbox(label="Status", lines=1, interactive=False)

                with gr.Row():
                    game_type = gr.Radio(
                        choices=["snake", "pacman", "dungeon", "local_view"],
                        value="pacman",
                        label="🎮 Game"
                    )
                    difficulty = gr.Slider(0, 2, value=1, step=1, label="Difficulty")

                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset", variant="secondary", size="lg")
                    step_btn = gr.Button("▶️ Step", variant="primary", size="lg")
                    run_btn = gr.Button("🎬 Auto-Play", variant="stop", size="lg")

            # Right: Stats
            with gr.Column(scale=1):
                stats_panel = gr.HTML(label="📊 Statistics")

                gr.Markdown("""
                ### 🎯 Zero-Shot Transfer

                **Training:** Snake only
                **Testing:** All 4 games

                **Results:**
                - 👾 Pac-Man: 75% win
                - 🏰 Dungeon: 67% win
                - ✈️ Sky: Testing

                ### 🎨 Features
                - Smooth Pygame graphics
                - Real-time AI decisions
                - 16 raycast sensors
                - Context-aware DQN
                """)

        # Event handlers
        def reset_wrapper(game, diff):
            global demo_instance
            if not demo_instance:
                demo_instance = PygameDemo()
            return demo_instance.reset_game(game, int(diff))

        reset_btn.click(reset_wrapper, inputs=[game_type, difficulty],
                       outputs=[game_display, stats_panel])

        def step_wrapper(game):
            if demo_instance:
                return demo_instance.step(game)
            return None, None, "Reset first"

        step_btn.click(step_wrapper, inputs=[game_type],
                      outputs=[game_display, stats_panel, status_text])

        def auto_play(game, diff):
            global demo_instance
            if not demo_instance:
                demo_instance = PygameDemo()

            img, stats = demo_instance.reset_game(game, int(diff))
            yield img, stats, "Starting..."

            import time
            for step in range(500):
                if demo_instance.stats['done']:
                    break

                img, stats, status = demo_instance.step(game)
                if step % 2 == 0:  # Update every 2 steps
                    yield img, stats, status
                    time.sleep(0.05)

            yield img, stats, "Episode Complete"

        run_btn.click(auto_play, inputs=[game_type, difficulty],
                     outputs=[game_display, stats_panel, status_text])

        # Initialize
        demo.load(reset_wrapper, inputs=[gr.State("pacman"), gr.State(1)],
                 outputs=[game_display, stats_panel])

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)

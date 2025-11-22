"""
Multi-Game Foundation Agent Demo - Side-by-Side Human vs AI

Optimized layout:
- Top: Game selection
- Left: Parameters and controls
- Center: Human game | AI game (side-by-side)
- Right: Score panel

Features continuous auto-play mode for both human and AI.
"""

import gradio as gr
import numpy as np
import torch
import sys
import os
from PIL import Image, ImageDraw, ImageFont
import time
import threading

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "core"))

from src.context_aware_agent import ContextAwareDQN, add_context_to_observation
from src.core.expanded_temporal_observer import ExpandedTemporalObserver
from src.core.enhanced_snake_game import EnhancedSnakeGame
from src.core.simple_pacman_game import SimplePacManGame
from src.core.simple_dungeon_game import SimpleDungeonGame
from src.core.local_view_game import LocalViewGame

# Colors
COLORS = {
    'bg_dark': (25, 28, 35),
    'bg_grid': (35, 38, 45),
    'snake_head': (34, 197, 94),
    'snake_body': (74, 222, 128),
    'snake_food': (239, 68, 68),
    'pacman': (255, 215, 0),
    'pellet': (255, 255, 255),
    'ghost_red': (255, 100, 100),
    'ghost_pink': (255, 184, 255),
    'ghost_orange': (255, 184, 82),
    'ghost_cyan': (0, 255, 255),
    'player': (34, 197, 94),
    'treasure': (255, 215, 0),
    'enemy_purple': (168, 85, 247),
    'enemy_red': (239, 68, 68),
    'enemy_orange': (249, 115, 22),
    'airplane': (59, 130, 246),
    'cloud_coin': (255, 215, 0),
    'bird_enemy': (239, 68, 68),
    'sky': (135, 206, 250),
    'wall': (100, 116, 139),
    'wall_dark': (51, 65, 85),
    'grid_line': (55, 58, 65),
    'text_white': (255, 255, 255),
    'accent_blue': (59, 130, 246),
    'accent_green': (34, 197, 94),
}


class GameRenderer:
    """Unified renderer for all games"""

    @staticmethod
    def render_snake(game, cell_size=25):
        """Render Snake game with 90s vibe"""
        size = game.size
        img = Image.new('RGB', (size * cell_size, size * cell_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw boundary walls with dark green
        dark_green = (0, 100, 0)
        for i in range(size):
            draw.rectangle([i * cell_size, 0, (i+1) * cell_size, cell_size], fill=dark_green)
            draw.rectangle([i * cell_size, (size-1) * cell_size, (i+1) * cell_size, size * cell_size], fill=dark_green)
            draw.rectangle([0, i * cell_size, cell_size, (i+1) * cell_size], fill=dark_green)
            draw.rectangle([(size-1) * cell_size, i * cell_size, size * cell_size, (i+1) * cell_size], fill=dark_green)

        # Draw obstacles (gray blocks)
        if hasattr(game, 'central_obstacles'):
            for ox, oy in game.central_obstacles:
                draw.rectangle([ox * cell_size, oy * cell_size,
                              (ox + 1) * cell_size, (oy + 1) * cell_size],
                             fill=(128, 128, 128))

        # Food (red circles)
        if hasattr(game, 'food_positions'):
            for fx, fy in game.food_positions:
                cx, cy = int((fx + 0.5) * cell_size), int((fy + 0.5) * cell_size)
                r = int(cell_size * 0.4)
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 0, 0))

        # Snake body (green with inset)
        if hasattr(game, 'snake') and game.snake:
            for x, y in game.snake[1:]:
                draw.rectangle([x * cell_size + 2, y * cell_size + 2,
                              (x + 1) * cell_size - 2, (y + 1) * cell_size - 2],
                             fill=(0, 255, 0))

            # Snake head (cyan with inset)
            if game.snake:
                hx, hy = game.snake[0]
                draw.rectangle([hx * cell_size + 1, hy * cell_size + 1,
                              (hx + 1) * cell_size - 1, (hy + 1) * cell_size - 1],
                             fill=(0, 255, 255))
        return img

    @staticmethod
    def render_pacman(game, cell_size=25):
        """Render Pac-Man game with 90s vibe"""
        size = game.size
        img = Image.new('RGB', (size * cell_size, size * cell_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Walls (blue blocks)
        for wx, wy in game.walls:
            draw.rectangle([wx * cell_size, wy * cell_size,
                          (wx + 1) * cell_size, (wy + 1) * cell_size],
                         fill=(0, 0, 255))

        # Pellets (small white circles)
        for px, py in game.pellets:
            cx, cy = int((px + 0.5) * cell_size), int((py + 0.5) * cell_size)
            draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(255, 255, 255))

        # Ghosts with eyes
        ghost_colors = [(255, 0, 0), (255, 192, 203), (255, 165, 0)]  # RED, PINK, ORANGE
        for i, ghost in enumerate(game.ghosts):
            gx, gy = ghost['pos']
            cx, cy = int((gx + 0.5) * cell_size), int((gy + 0.5) * cell_size)
            r = int(cell_size * 0.4)
            # Ghost body
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=ghost_colors[i % len(ghost_colors)])
            # Eyes
            eye_left_x = int((gx + 0.3) * cell_size)
            eye_right_x = int((gx + 0.7) * cell_size)
            eye_y = int((gy + 0.4) * cell_size)
            draw.ellipse([eye_left_x-2, eye_y-2, eye_left_x+2, eye_y+2], fill=(255, 255, 255))
            draw.ellipse([eye_right_x-2, eye_y-2, eye_right_x+2, eye_y+2], fill=(255, 255, 255))

        # Pac-Man (yellow with mouth)
        pmx, pmy = game.pacman_pos
        cx, cy = int((pmx + 0.5) * cell_size), int((pmy + 0.5) * cell_size)
        r = int(cell_size * 0.45)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 255, 0))

        return img

    @staticmethod
    def render_dungeon(game, cell_size=25):
        """Render Dungeon game with 90s vibe"""
        size = game.size
        img = Image.new('RGB', (size * cell_size, size * cell_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Walls with 3D border effect
        dark_gray = (64, 64, 64)
        light_gray = (128, 128, 128)
        for wx, wy in game.walls:
            # Main wall
            draw.rectangle([wx * cell_size, wy * cell_size,
                          (wx + 1) * cell_size, (wy + 1) * cell_size],
                         fill=dark_gray)
            # Border for depth
            draw.rectangle([wx * cell_size, wy * cell_size,
                          (wx + 1) * cell_size, (wy + 1) * cell_size],
                         outline=light_gray, width=1)

        # Treasures with glow effect
        for tx, ty in game.treasures:
            cx, cy = int((tx + 0.5) * cell_size), int((ty + 0.5) * cell_size)
            # Outer glow (yellow)
            draw.ellipse([cx-8, cy-8, cx+8, cy+8], fill=(255, 255, 0))
            # Inner gold
            draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill=(255, 215, 0))

        # Enemies with eyes
        enemy_colors = [(255, 0, 0), (128, 0, 128), (255, 165, 0)]  # RED, PURPLE, ORANGE
        for i, enemy in enumerate(game.enemies):
            ex, ey = enemy['pos']
            # Enemy body
            draw.rectangle([ex * cell_size + 2, ey * cell_size + 2,
                          (ex + 1) * cell_size - 2, (ey + 1) * cell_size - 2],
                         fill=enemy_colors[i % len(enemy_colors)])
            # Eyes
            eye_left_x = int((ex + 0.3) * cell_size)
            eye_right_x = int((ex + 0.7) * cell_size)
            eye_y = int((ey + 0.4) * cell_size)
            draw.ellipse([eye_left_x-3, eye_y-3, eye_left_x+3, eye_y+3], fill=(255, 255, 255))
            draw.ellipse([eye_right_x-3, eye_y-3, eye_right_x+3, eye_y+3], fill=(255, 255, 255))

        # Player (green with cyan highlight)
        px, py = game.player_pos
        cx, cy = int((px + 0.5) * cell_size), int((py + 0.5) * cell_size)
        r = int(cell_size * 0.4)
        # Body
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(0, 255, 0))
        # Highlight border
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(0, 255, 255), width=2)

        return img

    @staticmethod
    def render_local_view(game, cell_size=12, viewport_size=25):
        """Render Sky Collector (airplane theme) with 90s vibe"""
        # Sky blue background
        img = Image.new('RGB', (viewport_size * cell_size, viewport_size * cell_size), (135, 206, 250))
        draw = ImageDraw.Draw(img)

        ax, ay = game.agent_pos
        half_view = viewport_size // 2
        view_x, view_y = ax - half_view, ay - half_view

        def to_screen(wx, wy):
            return (wx - view_x) * cell_size, (wy - view_y) * cell_size

        def in_view(wx, wy):
            return view_x <= wx < view_x + viewport_size and view_y <= wy < view_y + viewport_size

        # Draw grid (subtle)
        grid_color = (100, 150, 200)
        for i in range(viewport_size + 1):
            draw.line([(i * cell_size, 0), (i * cell_size, viewport_size * cell_size)],
                     fill=grid_color, width=1)
            draw.line([(0, i * cell_size), (viewport_size * cell_size, i * cell_size)],
                     fill=grid_color, width=1)

        # Walls (clouds - blue blocks)
        for wx, wy in game.walls:
            if in_view(wx, wy):
                sx, sy = to_screen(wx, wy)
                draw.rectangle([sx, sy, sx + cell_size, sy + cell_size], fill=(0, 100, 255))

        # Coins with glow effect
        for cx, cy in game.coins:
            if in_view(cx, cy):
                sx, sy = to_screen(cx, cy)
                center_x, center_y = int(sx + cell_size/2), int(sy + cell_size/2)
                # Outer glow (yellow)
                draw.ellipse([center_x-5, center_y-5, center_x+5, center_y+5], fill=(255, 255, 0))
                # Inner coin (gold)
                draw.ellipse([center_x-3, center_y-3, center_x+3, center_y+3], fill=(255, 215, 0))

        # Enemies (birds) with detail
        enemy_colors = [(255, 0, 0), (255, 165, 0), (200, 0, 255)]  # RED, ORANGE, PURPLE
        for i, enemy in enumerate(game.enemies):
            ex, ey = enemy['pos']
            if in_view(ex, ey):
                sx, sy = to_screen(ex, ey)
                center_x, center_y = int(sx + cell_size/2), int(sy + cell_size/2)
                color = enemy_colors[i % len(enemy_colors)]
                # Body
                draw.rectangle([sx + 2, sy + 2, sx + cell_size - 2, sy + cell_size - 2],
                             fill=color)
                # Eyes
                eye_left_x = int(sx + cell_size * 0.35)
                eye_right_x = int(sx + cell_size * 0.65)
                eye_y = int(sy + cell_size * 0.4)
                draw.ellipse([eye_left_x-2, eye_y-2, eye_left_x+2, eye_y+2], fill=(255, 255, 255))
                draw.ellipse([eye_right_x-2, eye_y-2, eye_right_x+2, eye_y+2], fill=(255, 255, 255))

        # Airplane (agent) with glow highlight
        sx, sy = to_screen(ax, ay)
        cx, cy = int(sx + cell_size/2), int(sy + cell_size/2)
        r = int(cell_size * 0.5)
        # Glow/highlight
        draw.ellipse([cx-r-2, cy-r-2, cx+r+2, cy+r+2], outline=(0, 255, 255), width=2)
        # Body
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(0, 255, 0))

        return img


class MultiGameDemo:
    """Manages side-by-side gameplay with auto-play"""

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.agent = None
        self.observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)
        self.human_game = None
        self.ai_game = None
        self.human_stats = {'score': 0, 'steps': 0, 'done': False}
        self.ai_stats = {'score': 0, 'steps': 0, 'done': False}
        self.playing_human = False
        self.playing_ai = False
        self.load_model()

    def load_model(self):
        """Load trained agent"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                self.agent = ContextAwareDQN(obs_dim=183, action_dim=4)
                self.agent.load_state_dict(checkpoint['policy_net'])
                self.agent.eval()
                return f"✓ Model loaded successfully"
            except Exception as e:
                return f"✗ Error: {e}"
        return "⚠ No model - AI uses random actions"

    def reset_games(self, game_type, difficulty=0):
        """Reset both games"""
        self.human_stats = {'score': 0, 'steps': 0, 'done': False}
        self.ai_stats = {'score': 0, 'steps': 0, 'done': False}
        self.playing_human = False
        self.playing_ai = False

        if game_type == 'snake':
            size = [10, 15, 20][difficulty]
            pellets = [3, 5, 7][difficulty]
            self.human_game = EnhancedSnakeGame(size=size, initial_pellets=pellets, max_steps=400)
            self.ai_game = EnhancedSnakeGame(size=size, initial_pellets=pellets, max_steps=400)
        elif game_type == 'pacman':
            self.human_game = SimplePacManGame(size=20, num_pellets=30, ghost_level=difficulty, max_steps=500)
            self.ai_game = SimplePacManGame(size=20, num_pellets=30, ghost_level=difficulty, max_steps=500)
        elif game_type == 'dungeon':
            self.human_game = SimpleDungeonGame(size=20, num_treasures=3, enemy_level=difficulty, max_steps=600)
            self.ai_game = SimpleDungeonGame(size=20, num_treasures=3, enemy_level=difficulty, max_steps=600)
        elif game_type == 'local_view':
            self.human_game = LocalViewGame(world_size=40, num_coins=20, enemy_level=difficulty, max_steps=1000)
            self.ai_game = LocalViewGame(world_size=40, num_coins=20, enemy_level=difficulty, max_steps=1000)

        self.human_game.reset()
        self.ai_game.reset()
        self.observer.reset()

        return self.render_games(game_type), self.get_stats_html(game_type)

    def step_game(self, game, action):
        """Execute single step"""
        _, reward, done = game.step(action)
        return done

    def get_ai_action(self, game):
        """Get action from AI agent"""
        if not self.agent:
            return np.random.randint(4)

        state = self._get_state_dict(game)
        obs = self.observer.observe(state)
        context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        obs_with_context = add_context_to_observation(obs, context)
        return self.agent.get_action(obs_with_context, epsilon=0.0)

    def _get_state_dict(self, game):
        """Convert game to state dict"""
        state = {
            'grid_size': (getattr(game, 'size', getattr(game, 'world_size', 20)),) * 2,
            'walls': game.walls,
            'score': game.score,
            'done': False
        }

        if hasattr(game, 'snake'):
            state['agent_pos'] = game.snake[0]
            state['snake_body'] = game.snake[1:]
            state['rewards'] = list(game.food_positions)
        elif hasattr(game, 'pacman_pos'):
            state['agent_pos'] = game.pacman_pos
            state['rewards'] = list(game.pellets)
            state['entities'] = [{'pos': g['pos']} for g in game.ghosts]
        elif hasattr(game, 'player_pos'):
            state['agent_pos'] = game.player_pos
            state['rewards'] = list(game.treasures)
            state['entities'] = [{'pos': e['pos']} for e in game.enemies]
        elif hasattr(game, 'agent_pos'):
            state['agent_pos'] = game.agent_pos
            state['rewards'] = list(game.coins)
            state['entities'] = [{'pos': e['pos']} for e in game.enemies]

        return state

    def render_games(self, game_type):
        """Render both games side-by-side"""
        if game_type == 'snake':
            h_img = GameRenderer.render_snake(self.human_game, cell_size=35)
            a_img = GameRenderer.render_snake(self.ai_game, cell_size=35)
        elif game_type == 'pacman':
            h_img = GameRenderer.render_pacman(self.human_game, cell_size=35)
            a_img = GameRenderer.render_pacman(self.ai_game, cell_size=35)
        elif game_type == 'dungeon':
            h_img = GameRenderer.render_dungeon(self.human_game, cell_size=35)
            a_img = GameRenderer.render_dungeon(self.ai_game, cell_size=35)
        elif game_type == 'local_view':
            h_img = GameRenderer.render_local_view(self.human_game, cell_size=20)
            a_img = GameRenderer.render_local_view(self.ai_game, cell_size=20)

        # Combine side-by-side with nice spacing
        gap = 30
        combined = Image.new('RGB', (h_img.width + a_img.width + gap, h_img.height), (0, 0, 0))
        combined.paste(h_img, (0, 0))
        combined.paste(a_img, (h_img.width + gap, 0))

        # Add separator line with labels
        draw = ImageDraw.Draw(combined)
        mid_x = h_img.width + gap // 2
        draw.line([(mid_x, 0), (mid_x, h_img.height)], fill=(100, 100, 100), width=3)

        return combined

    def get_stats_html(self, game_type):
        """Generate stats HTML"""
        self.human_stats['score'] = self.human_game.score if self.human_game else 0
        self.ai_stats['score'] = self.ai_game.score if self.ai_game else 0

        game_names = {'snake': '🐍 Snake', 'pacman': '👾 Pac-Man',
                      'dungeon': '🏰 Dungeon', 'local_view': '✈️ Sky Collector'}

        html = f"""
        <div style="font-family: 'Segoe UI', Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 25px; border-radius: 15px; color: white; height: 100%;">
            <h2 style="text-align: center; margin-bottom: 25px; font-size: 26px;">
                {game_names.get(game_type, 'Game')}
            </h2>

            <div style="background: rgba(59, 130, 246, 0.3); padding: 20px; border-radius: 12px; margin-bottom: 15px;">
                <h3 style="color: #60a5fa; margin-bottom: 12px; font-size: 18px;">🎮 HUMAN</h3>
                <p style="font-size: 16px; margin: 8px 0;"><strong>Score:</strong>
                   <span style="color: #fbbf24; font-size: 28px; font-weight: bold;">{self.human_stats['score']}</span></p>
                <p style="font-size: 14px; margin: 5px 0;">Steps: {self.human_stats['steps']}</p>
                <p style="font-size: 14px; margin: 5px 0;">Status:
                   <span style="color: {'#ef4444' if self.human_stats['done'] else '#10b981'};">
                   {'Game Over' if self.human_stats['done'] else 'Playing'}</span></p>
            </div>

            <div style="background: rgba(34, 197, 94, 0.3); padding: 20px; border-radius: 12px; margin-bottom: 15px;">
                <h3 style="color: #4ade80; margin-bottom: 12px; font-size: 18px;">🤖 AI AGENT</h3>
                <p style="font-size: 16px; margin: 8px 0;"><strong>Score:</strong>
                   <span style="color: #fbbf24; font-size: 28px; font-weight: bold;">{self.ai_stats['score']}</span></p>
                <p style="font-size: 14px; margin: 5px 0;">Steps: {self.ai_stats['steps']}</p>
                <p style="font-size: 14px; margin: 5px 0;">Status:
                   <span style="color: {'#ef4444' if self.ai_stats['done'] else '#10b981'};">
                   {'Game Over' if self.ai_stats['done'] else 'Playing'}</span></p>
            </div>

            <div style="background: rgba(251, 191, 36, 0.2); padding: 20px; border-radius: 12px; text-align: center;">
                <h3 style="color: #fbbf24; margin-bottom: 12px; font-size: 18px;">📊 Score</h3>
                <div style="font-size: 36px; font-weight: bold; margin: 15px 0;">
                    {self.human_stats['score']} <span style="color: #9ca3af; font-size: 24px;">vs</span> {self.ai_stats['score']}
                </div>
                <div style="font-size: 18px; color: #fbbf24;">
                    {"🏆 Human Wins!" if self.human_stats['score'] > self.ai_stats['score'] else
                     "🤖 AI Wins!" if self.ai_stats['score'] > self.human_stats['score'] else
                     "🤝 Tie!"}
                </div>
            </div>
        </div>
        """
        return html


# Global instance
demo_instance = None


def create_demo():
    """Create Gradio interface with optimized layout"""
    global demo_instance

    with gr.Blocks(title="Multi-Game Foundation Agent",
                   theme=gr.themes.Soft(primary_hue="blue")) as demo:

        gr.Markdown("""
        # 🎮 Multi-Game Foundation Agent: Human vs AI
        **Side-by-side gameplay showcase** - Watch AI trained on Snake play all 4 games!
        """)

        # Top: Game selection
        with gr.Row():
            game_type = gr.Radio(
                choices=["snake", "pacman", "dungeon", "local_view"],
                value="pacman",
                label="🎮 Select Game",
                scale=2
            )
            difficulty = gr.Slider(0, 2, value=0, step=1,
                                  label="🎚️ Difficulty",
                                  info="0=Easy, 1=Medium, 2=Hard",
                                  scale=1)
            model_path = gr.Textbox(
                value="checkpoints/multi_game_enhanced_*_policy.pth",
                label="🤖 Model Path",
                scale=2
            )

        # Main layout: Parameters | Games | Scores
        with gr.Row():
            # Left: Parameters and Controls
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Controls")

                with gr.Group():
                    load_btn = gr.Button("📥 Load Model", variant="secondary")
                    reset_btn = gr.Button("🔄 Reset Games", variant="secondary", size="lg")
                    model_status = gr.Textbox(label="Status", lines=2, interactive=False)

                gr.Markdown("### 🎮 Human Controls")
                with gr.Column():
                    up_btn = gr.Button("⬆️ UP", size="sm")
                    with gr.Row():
                        left_btn = gr.Button("⬅️", size="sm", scale=1)
                        down_btn = gr.Button("⬇️", size="sm", scale=1)
                        right_btn = gr.Button("➡️", size="sm", scale=1)
                    play_human_btn = gr.Button("▶️ Auto-Play Human", variant="primary")

                gr.Markdown("### 🤖 AI Controls")
                play_ai_btn = gr.Button("▶️ Auto-Play AI", variant="primary", size="lg")
                play_both_btn = gr.Button("⚡ Auto-Play BOTH", variant="stop", size="lg")

                gr.Markdown("""
                **Auto-Play:**
                - Starts from beginning
                - Runs until game over
                - Real-time visualization
                - Automatic scoring
                """)

            # Center: Both games
            with gr.Column(scale=3):
                game_display = gr.Image(label="🎮 Human (Left) vs 🤖 AI (Right)", type="pil", height=750)

            # Right: Scores
            with gr.Column(scale=1):
                stats_panel = gr.HTML(label="📊 Statistics")

        # Event handlers
        def load_model_wrapper(model_pattern):
            global demo_instance
            import glob
            files = glob.glob(model_pattern)
            if files:
                model_file = max(files, key=os.path.getctime)
                demo_instance = MultiGameDemo(model_path=model_file)
                return demo_instance.load_model() + f"\n{os.path.basename(model_file)}"
            demo_instance = MultiGameDemo()
            return f"✗ No files matching: {model_pattern}"

        load_btn.click(load_model_wrapper, inputs=[model_path], outputs=[model_status])

        def reset_wrapper(game, diff):
            global demo_instance
            if not demo_instance:
                demo_instance = MultiGameDemo()
            return demo_instance.reset_games(game, int(diff))

        reset_btn.click(reset_wrapper, inputs=[game_type, difficulty],
                       outputs=[game_display, stats_panel])

        # Human controls
        def human_step(action, game):
            if demo_instance and not demo_instance.human_stats['done']:
                demo_instance.step_game(demo_instance.human_game, action)
                demo_instance.human_stats['steps'] += 1
                demo_instance.human_stats['done'] = demo_instance.human_game.score >= getattr(
                    demo_instance.human_game, 'num_pellets',
                    getattr(demo_instance.human_game, 'num_treasures',
                           getattr(demo_instance.human_game, 'num_coins', 999)))
                return demo_instance.render_games(game), demo_instance.get_stats_html(game)
            return None, "Reset game first"

        up_btn.click(lambda g: human_step(0, g), inputs=[game_type], outputs=[game_display, stats_panel])
        down_btn.click(lambda g: human_step(1, g), inputs=[game_type], outputs=[game_display, stats_panel])
        left_btn.click(lambda g: human_step(2, g), inputs=[game_type], outputs=[game_display, stats_panel])
        right_btn.click(lambda g: human_step(3, g), inputs=[game_type], outputs=[game_display, stats_panel])

        # Auto-play functions
        def auto_play_human(game):
            """Auto-play human with random actions"""
            if not demo_instance:
                return None, "Reset first"

            for step in range(500):
                if demo_instance.human_stats['done']:
                    break

                action = np.random.randint(4)
                demo_instance.step_game(demo_instance.human_game, action)
                demo_instance.human_stats['steps'] += 1

                if step % 2 == 0:  # Update every 2 steps for smooth animation
                    yield demo_instance.render_games(game), demo_instance.get_stats_html(game)
                    time.sleep(0.03)

            demo_instance.human_stats['done'] = True
            yield demo_instance.render_games(game), demo_instance.get_stats_html(game)

        def auto_play_ai(game):
            """Auto-play AI agent"""
            if not demo_instance:
                return None, "Reset first"

            for step in range(500):
                if demo_instance.ai_stats['done']:
                    break

                action = demo_instance.get_ai_action(demo_instance.ai_game)
                demo_instance.step_game(demo_instance.ai_game, action)
                demo_instance.ai_stats['steps'] += 1

                if step % 2 == 0:
                    yield demo_instance.render_games(game), demo_instance.get_stats_html(game)
                    time.sleep(0.03)

            demo_instance.ai_stats['done'] = True
            yield demo_instance.render_games(game), demo_instance.get_stats_html(game)

        def auto_play_both(game):
            """Auto-play both simultaneously"""
            if not demo_instance:
                return None, "Reset first"

            for step in range(500):
                if demo_instance.human_stats['done'] and demo_instance.ai_stats['done']:
                    break

                # Human step (random)
                if not demo_instance.human_stats['done']:
                    h_action = np.random.randint(4)
                    demo_instance.step_game(demo_instance.human_game, h_action)
                    demo_instance.human_stats['steps'] += 1

                # AI step
                if not demo_instance.ai_stats['done']:
                    a_action = demo_instance.get_ai_action(demo_instance.ai_game)
                    demo_instance.step_game(demo_instance.ai_game, a_action)
                    demo_instance.ai_stats['steps'] += 1

                if step % 2 == 0:
                    yield demo_instance.render_games(game), demo_instance.get_stats_html(game)
                    time.sleep(0.03)

            demo_instance.human_stats['done'] = True
            demo_instance.ai_stats['done'] = True
            yield demo_instance.render_games(game), demo_instance.get_stats_html(game)

        play_human_btn.click(auto_play_human, inputs=[game_type],
                            outputs=[game_display, stats_panel])
        play_ai_btn.click(auto_play_ai, inputs=[game_type],
                         outputs=[game_display, stats_panel])
        play_both_btn.click(auto_play_both, inputs=[game_type],
                           outputs=[game_display, stats_panel])

        # Initialize
        demo.load(reset_wrapper, inputs=[game_type, difficulty],
                 outputs=[game_display, stats_panel])

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)

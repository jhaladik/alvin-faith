"""
Smooth Arcade Demo - Professional HTML-style Interface
Matches the look and feel of index.html with smooth graphics
"""

import gradio as gr
import numpy as np
import torch
import sys
import os
from PIL import Image, ImageDraw, ImageFont
import time

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "core"))

from src.context_aware_agent import ContextAwareDQN, add_context_to_observation
from src.core.expanded_temporal_observer import ExpandedTemporalObserver
from src.core.enhanced_snake_game import EnhancedSnakeGame
from src.core.simple_pacman_game import SimplePacManGame
from src.core.simple_dungeon_game import SimpleDungeonGame
from src.core.local_view_game import LocalViewGame


class SmoothRenderer:
    """Smooth, anti-aliased renderer like HTML canvas"""

    @staticmethod
    def draw_circle(draw, x, y, radius, fill_color, outline_color=None, outline_width=0):
        """Draw anti-aliased circle"""
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, fill=fill_color, outline=outline_color, width=outline_width)

    @staticmethod
    def draw_rounded_rect(draw, x, y, w, h, radius, fill_color, outline_color=None, outline_width=0):
        """Draw rounded rectangle"""
        draw.rounded_rectangle([x, y, x + w, y + h], radius=radius,
                              fill=fill_color, outline=outline_color, width=outline_width)

    @staticmethod
    def render_snake(game, cell_size=30):
        """Smooth Snake rendering"""
        size = game.size
        img = Image.new('RGB', (size * cell_size, size * cell_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw boundary walls
        wall_color = (0, 100, 0)
        for i in range(size):
            # Top/bottom walls
            SmoothRenderer.draw_rounded_rect(draw, i * cell_size, 0, cell_size, cell_size,
                                            3, wall_color, (0, 200, 0), 1)
            SmoothRenderer.draw_rounded_rect(draw, i * cell_size, (size-1) * cell_size,
                                            cell_size, cell_size, 3, wall_color, (0, 200, 0), 1)
            # Left/right walls
            SmoothRenderer.draw_rounded_rect(draw, 0, i * cell_size, cell_size, cell_size,
                                            3, wall_color, (0, 200, 0), 1)
            SmoothRenderer.draw_rounded_rect(draw, (size-1) * cell_size, i * cell_size,
                                            cell_size, cell_size, 3, wall_color, (0, 200, 0), 1)

        # Obstacles
        if hasattr(game, 'central_obstacles'):
            for ox, oy in game.central_obstacles:
                SmoothRenderer.draw_rounded_rect(draw, ox * cell_size + 2, oy * cell_size + 2,
                                                cell_size - 4, cell_size - 4,
                                                5, (80, 80, 80), (150, 150, 150), 2)

        # Food - smooth circles
        if hasattr(game, 'food_positions'):
            for fx, fy in game.food_positions:
                cx = (fx + 0.5) * cell_size
                cy = (fy + 0.5) * cell_size
                # Outer glow
                SmoothRenderer.draw_circle(draw, cx, cy, cell_size * 0.45, (200, 0, 0))
                # Inner core
                SmoothRenderer.draw_circle(draw, cx, cy, cell_size * 0.35, (255, 0, 0))

        # Snake body - smooth segments
        if hasattr(game, 'snake') and game.snake:
            for i, (x, y) in enumerate(game.snake[1:]):
                cx = (x + 0.5) * cell_size
                cy = (y + 0.5) * cell_size
                # Gradient effect
                SmoothRenderer.draw_circle(draw, cx, cy, cell_size * 0.42, (0, 200, 0))
                SmoothRenderer.draw_circle(draw, cx, cy, cell_size * 0.35, (0, 255, 0))

            # Snake head - cyan with highlight
            if game.snake:
                hx, hy = game.snake[0]
                cx = (hx + 0.5) * cell_size
                cy = (hy + 0.5) * cell_size
                SmoothRenderer.draw_circle(draw, cx, cy, cell_size * 0.45, (0, 200, 200))
                SmoothRenderer.draw_circle(draw, cx, cy, cell_size * 0.38, (0, 255, 255))
                # Eye highlight
                SmoothRenderer.draw_circle(draw, cx - 4, cy - 4, 3, (255, 255, 255))

        return img

    @staticmethod
    def render_pacman(game, cell_size=30):
        """Smooth Pac-Man rendering"""
        size = game.size
        img = Image.new('RGB', (size * cell_size, size * cell_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Walls - smooth blue blocks
        for wx, wy in game.walls:
            SmoothRenderer.draw_rounded_rect(draw, wx * cell_size + 1, wy * cell_size + 1,
                                            cell_size - 2, cell_size - 2,
                                            4, (0, 0, 180), (0, 100, 255), 2)

        # Pellets - small white circles
        for px, py in game.pellets:
            cx = (px + 0.5) * cell_size
            cy = (py + 0.5) * cell_size
            SmoothRenderer.draw_circle(draw, cx, cy, 4, (255, 255, 255))

        # Ghosts - smooth with eyes
        ghost_colors = [(255, 0, 0), (255, 184, 255), (255, 165, 0), (0, 255, 255)]
        for i, ghost in enumerate(game.ghosts):
            gx, gy = ghost['pos']
            cx = (gx + 0.5) * cell_size
            cy = (gy + 0.5) * cell_size
            r = cell_size * 0.42

            # Ghost body
            color = ghost_colors[i % len(ghost_colors)]
            SmoothRenderer.draw_circle(draw, cx, cy, r, color)

            # Eyes - white circles with pupils
            eye_r = 4
            pupil_r = 2
            # Left eye
            SmoothRenderer.draw_circle(draw, cx - 6, cy - 3, eye_r, (255, 255, 255))
            SmoothRenderer.draw_circle(draw, cx - 6, cy - 3, pupil_r, (0, 0, 150))
            # Right eye
            SmoothRenderer.draw_circle(draw, cx + 6, cy - 3, eye_r, (255, 255, 255))
            SmoothRenderer.draw_circle(draw, cx + 6, cy - 3, pupil_r, (0, 0, 150))

        # Pac-Man - smooth yellow circle
        pmx, pmy = game.pacman_pos
        cx = (pmx + 0.5) * cell_size
        cy = (pmy + 0.5) * cell_size
        r = cell_size * 0.45
        SmoothRenderer.draw_circle(draw, cx, cy, r, (255, 255, 0), (255, 200, 0), 2)

        return img

    @staticmethod
    def render_dungeon(game, cell_size=30):
        """Smooth Dungeon rendering"""
        size = game.size
        img = Image.new('RGB', (size * cell_size, size * cell_size), (20, 20, 30))
        draw = ImageDraw.Draw(img)

        # Walls - 3D stone effect
        for wx, wy in game.walls:
            # Base
            SmoothRenderer.draw_rounded_rect(draw, wx * cell_size + 1, wy * cell_size + 1,
                                            cell_size - 2, cell_size - 2,
                                            3, (60, 60, 70), (100, 100, 110), 1)
            # Highlight
            draw.line([wx * cell_size + 2, wy * cell_size + 2,
                      (wx + 1) * cell_size - 2, wy * cell_size + 2],
                     fill=(140, 140, 150), width=2)

        # Treasures - glowing diamonds
        for tx, ty in game.treasures:
            cx = (tx + 0.5) * cell_size
            cy = (ty + 0.5) * cell_size
            # Glow
            SmoothRenderer.draw_circle(draw, cx, cy, 12, (200, 180, 0))
            # Diamond shape
            points = [(cx, cy - 10), (cx + 10, cy), (cx, cy + 10), (cx - 10, cy)]
            draw.polygon(points, fill=(255, 215, 0), outline=(255, 255, 100))

        # Enemies - smooth with eyes
        enemy_colors = [(255, 50, 50), (200, 50, 255), (255, 140, 0)]
        for i, enemy in enumerate(game.enemies):
            ex, ey = enemy['pos']
            cx = (ex + 0.5) * cell_size
            cy = (ey + 0.5) * cell_size
            r = cell_size * 0.4

            color = enemy_colors[i % len(enemy_colors)]
            SmoothRenderer.draw_circle(draw, cx, cy, r, color, (255, 255, 255), 1)

            # Eyes
            SmoothRenderer.draw_circle(draw, cx - 5, cy - 3, 3, (255, 255, 255))
            SmoothRenderer.draw_circle(draw, cx + 5, cy - 3, 3, (255, 255, 255))
            SmoothRenderer.draw_circle(draw, cx - 5, cy - 3, 2, (255, 0, 0))
            SmoothRenderer.draw_circle(draw, cx + 5, cy - 3, 2, (255, 0, 0))

        # Player - smooth with glow
        px, py = game.player_pos
        cx = (px + 0.5) * cell_size
        cy = (py + 0.5) * cell_size
        # Glow
        SmoothRenderer.draw_circle(draw, cx, cy, cell_size * 0.5, (0, 150, 0))
        # Body
        SmoothRenderer.draw_circle(draw, cx, cy, cell_size * 0.42, (0, 255, 0), (0, 255, 255), 2)
        # Highlight
        SmoothRenderer.draw_circle(draw, cx - 3, cy - 3, 3, (200, 255, 200))

        return img

    @staticmethod
    def render_local_view(game, cell_size=18, viewport_size=25):
        """Smooth Sky Collector rendering"""
        img = Image.new('RGB', (viewport_size * cell_size, viewport_size * cell_size),
                       (50, 100, 180))
        draw = ImageDraw.Draw(img)

        ax, ay = game.agent_pos
        half_view = viewport_size // 2
        view_x, view_y = ax - half_view, ay - half_view

        def to_screen(wx, wy):
            return (wx - view_x) * cell_size, (wy - view_y) * cell_size

        def in_view(wx, wy):
            return view_x <= wx < view_x + viewport_size and view_y <= wy < view_y + viewport_size

        # Walls (clouds) - smooth
        for wx, wy in game.walls:
            if in_view(wx, wy):
                sx, sy = to_screen(wx, wy)
                SmoothRenderer.draw_rounded_rect(draw, sx + 1, sy + 1, cell_size - 2, cell_size - 2,
                                                4, (100, 200, 255), (200, 230, 255), 1)

        # Coins - glowing
        for cx_pos, cy_pos in game.coins:
            if in_view(cx_pos, cy_pos):
                sx, sy = to_screen(cx_pos, cy_pos)
                cx = sx + cell_size / 2
                cy = sy + cell_size / 2
                # Glow
                SmoothRenderer.draw_circle(draw, cx, cy, 7, (200, 180, 0))
                # Coin
                SmoothRenderer.draw_circle(draw, cx, cy, 5, (255, 215, 0), (255, 255, 100), 1)

        # Enemies - smooth
        enemy_colors = [(255, 50, 50), (255, 140, 0), (200, 50, 255)]
        for i, enemy in enumerate(game.enemies):
            ex, ey = enemy['pos']
            if in_view(ex, ey):
                sx, sy = to_screen(ex, ey)
                cx = sx + cell_size / 2
                cy = sy + cell_size / 2
                color = enemy_colors[i % len(enemy_colors)]
                SmoothRenderer.draw_circle(draw, cx, cy, cell_size * 0.4, color, (255, 255, 255), 1)

                # Eyes
                SmoothRenderer.draw_circle(draw, cx - 3, cy - 2, 2, (255, 255, 255))
                SmoothRenderer.draw_circle(draw, cx + 3, cy - 2, 2, (255, 255, 255))

        # Airplane (player) - smooth with glow
        sx, sy = to_screen(ax, ay)
        cx = sx + cell_size / 2
        cy = sy + cell_size / 2
        # Glow
        SmoothRenderer.draw_circle(draw, cx, cy, cell_size * 0.5, (0, 200, 0))
        # Body
        SmoothRenderer.draw_circle(draw, cx, cy, cell_size * 0.42, (0, 255, 0), (0, 255, 255), 2)

        return img


class SmoothArcadeDemo:
    """Smooth arcade demo matching HTML style"""

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.agent = None
        self.observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)
        self.human_game = None
        self.ai_game = None
        self.human_stats = {'score': 0, 'steps': 0, 'done': False}
        self.ai_stats = {'score': 0, 'steps': 0, 'done': False}
        self.last_q_values = [0, 0, 0, 0]
        self.last_action_probs = [0.25, 0.25, 0.25, 0.25]
        self.action_history = []
        self.load_model()

    def load_model(self):
        """Load trained agent"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                self.agent = ContextAwareDQN(obs_dim=183, action_dim=4)
                self.agent.load_state_dict(checkpoint['policy_net'])
                self.agent.eval()
                return "✓ Model Loaded"
            except Exception as e:
                return f"✗ Error: {e}"
        return "⚠ Random Mode"

    def reset_games(self, game_type, difficulty=0):
        """Reset both games"""
        self.human_stats = {'score': 0, 'steps': 0, 'done': False}
        self.ai_stats = {'score': 0, 'steps': 0, 'done': False}
        self.action_history = []

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

        return self.render_games(game_type), self.get_stats_html(game_type), self.get_ai_metrics_html()

    def step_game(self, game, action):
        """Execute single step"""
        _, reward, done = game.step(action)
        return done

    def get_ai_action(self, game):
        """Get action from AI with Q-values"""
        if not self.agent:
            return np.random.randint(4)

        state = self._get_state_dict(game)
        obs = self.observer.observe(state)
        context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        obs_with_context = add_context_to_observation(obs, context)

        # Get Q-values
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_with_context).unsqueeze(0)
            q_values = self.agent.get_combined_q(obs_tensor).squeeze().numpy()
            self.last_q_values = q_values.tolist()

            # Action probabilities (softmax)
            exp_q = np.exp(q_values - np.max(q_values))
            self.last_action_probs = (exp_q / exp_q.sum()).tolist()

        action = np.argmax(q_values)
        self.action_history.append(action)
        if len(self.action_history) > 20:
            self.action_history.pop(0)

        return action

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
            h_img = SmoothRenderer.render_snake(self.human_game, cell_size=25)
            a_img = SmoothRenderer.render_snake(self.ai_game, cell_size=25)
        elif game_type == 'pacman':
            h_img = SmoothRenderer.render_pacman(self.human_game, cell_size=25)
            a_img = SmoothRenderer.render_pacman(self.ai_game, cell_size=25)
        elif game_type == 'dungeon':
            h_img = SmoothRenderer.render_dungeon(self.human_game, cell_size=25)
            a_img = SmoothRenderer.render_dungeon(self.ai_game, cell_size=25)
        elif game_type == 'local_view':
            h_img = SmoothRenderer.render_local_view(self.human_game, cell_size=16)
            a_img = SmoothRenderer.render_local_view(self.ai_game, cell_size=16)

        # Combine side-by-side
        gap = 20
        combined = Image.new('RGB', (h_img.width + a_img.width + gap, h_img.height), (10, 10, 10))
        combined.paste(h_img, (0, 0))
        combined.paste(a_img, (h_img.width + gap, 0))

        # Add labels
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        draw.text((h_img.width // 2, 10), "👤 HUMAN", fill=(255, 255, 0), anchor="mm", font=font)
        draw.text((h_img.width + gap + a_img.width // 2, 10), "🤖 AI", fill=(255, 0, 255), anchor="mm", font=font)

        return combined

    def get_stats_html(self, game_type):
        """Generate stats HTML matching index.html style"""
        self.human_stats['score'] = self.human_game.score if self.human_game else 0
        self.ai_stats['score'] = self.ai_game.score if self.ai_game else 0

        game_names = {'snake': '🐍 Snake', 'pacman': '👾 Pac-Man',
                      'dungeon': '🏰 Dungeon', 'local_view': '✈️ Sky Collector'}

        winner = ""
        if self.human_stats['score'] > self.ai_stats['score']:
            winner = "👤 Human Wins!"
        elif self.ai_stats['score'] > self.human_stats['score']:
            winner = "🤖 AI Wins!"
        else:
            winner = "🤝 Tie!"

        html = f"""
        <div style="font-family: 'Courier New', monospace; background: #0a0a0a; color: #fff;
                    padding: 15px; border: 2px solid #333; border-radius: 8px;">

            <h3 style="text-align: center; color: #ffff00; margin: 0 0 15px 0;">
                📊 Game Stats
            </h3>

            <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 8px;
                       padding: 12px; margin-bottom: 12px;">
                <h4 style="color: #ffff00; margin: 0 0 8px 0;">👤 Human Player</h4>
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span style="color: #aaa;">Score:</span>
                    <span style="color: #0f0; font-weight: bold;">{self.human_stats['score']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span style="color: #aaa;">Steps:</span>
                    <span style="color: #0f0; font-weight: bold;">{self.human_stats['steps']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span style="color: #aaa;">Status:</span>
                    <span style="color: {'#ff0000' if self.human_stats['done'] else '#00ff00'};">
                        {'GAME OVER' if self.human_stats['done'] else 'PLAYING'}
                    </span>
                </div>
            </div>

            <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 8px;
                       padding: 12px; margin-bottom: 12px;">
                <h4 style="color: #ff00ff; margin: 0 0 8px 0;">🤖 AI Avatar</h4>
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span style="color: #aaa;">Score:</span>
                    <span style="color: #0f0; font-weight: bold;">{self.ai_stats['score']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span style="color: #aaa;">Steps:</span>
                    <span style="color: #0f0; font-weight: bold;">{self.ai_stats['steps']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span style="color: #aaa;">Status:</span>
                    <span style="color: {'#ff0000' if self.ai_stats['done'] else '#00ff00'};">
                        {'GAME OVER' if self.ai_stats['done'] else 'PLAYING'}
                    </span>
                </div>
            </div>

            <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 8px;
                       padding: 12px; text-align: center;">
                <h4 style="color: #00ff00; margin: 0 0 8px 0;">⚖️ Comparison</h4>
                <div style="font-size: 24px; margin: 10px 0;">
                    <span style="color: #ffff00;">{self.human_stats['score']}</span>
                    <span style="color: #888;"> vs </span>
                    <span style="color: #ff00ff;">{self.ai_stats['score']}</span>
                </div>
                <div style="color: #ffff00; font-weight: bold; font-size: 14px;">
                    {winner}
                </div>
            </div>
        </div>
        """
        return html

    def get_ai_metrics_html(self):
        """Generate AI metrics HTML"""
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

        # Action distribution
        dist = [0, 0, 0, 0]
        if self.action_history:
            for a in self.action_history:
                dist[a] += 1
            dist = [d / len(self.action_history) * 100 for d in dist]

        html = f"""
        <div style="font-family: 'Courier New', monospace; background: #0a0a0a; color: #fff;
                    padding: 15px; border: 2px solid #333; border-radius: 8px;">

            <h3 style="text-align: center; color: #00ffff; margin: 0 0 15px 0;">
                🧠 AI Metrics
            </h3>

            <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 8px;
                       padding: 12px; margin-bottom: 12px;">
                <h4 style="color: #00ff00; margin: 0 0 8px 0;">⚡ DQN Q-Values</h4>
                <div style="font-size: 11px; color: #888; margin-bottom: 8px;">
                    State-Action Values:
                </div>
        """

        for i, (name, q_val) in enumerate(zip(action_names, self.last_q_values)):
            html += f"""
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span style="color: #aaa;">{name}:</span>
                    <span style="color: #ff00ff; font-family: 'Courier New'; font-weight: bold;">
                        {q_val:.4f}
                    </span>
                </div>
            """

        html += """
            </div>

            <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 8px;
                       padding: 12px; margin-bottom: 12px;">
                <h4 style="color: #ff66ff; margin: 0 0 8px 0;">🎯 Action Probabilities</h4>
        """

        for i, (name, prob) in enumerate(zip(action_names, self.last_action_probs)):
            html += f"""
                <div style="display: flex; justify-content: space-between; margin: 5px 0; align-items: center;">
                    <span style="color: #aaa; width: 60px;">{name}:</span>
                    <span style="color: #ff00ff; width: 50px; text-align: right;">{prob*100:.1f}%</span>
                    <div style="flex: 1; margin-left: 10px;">
                        <div style="width: 100%; height: 8px; background: #222; border-radius: 4px;">
                            <div style="width: {prob*100}%; height: 100%; background: linear-gradient(90deg, #00ff00, #00ffff);
                                       border-radius: 4px; transition: width 0.3s;"></div>
                        </div>
                    </div>
                </div>
            """

        html += """
            </div>

            <div style="background: #1a1a1a; border: 2px solid #333; border-radius: 8px;
                       padding: 12px;">
                <h4 style="color: #00ffff; margin: 0 0 8px 0;">📊 Action Distribution</h4>
                <div style="font-size: 11px; color: #888; margin-bottom: 8px;">
                    Last 20 AI Moves:
                </div>
        """

        for i, (name, pct) in enumerate(zip(action_names, dist)):
            html += f"""
                <div style="display: flex; justify-content: space-between; margin: 5px 0; align-items: center;">
                    <span style="color: #aaa; width: 60px;">{name}:</span>
                    <span style="color: #00ffff; width: 50px; text-align: right;">{pct:.1f}%</span>
                    <div style="flex: 1; margin-left: 10px;">
                        <div style="width: 100%; height: 6px; background: #222; border-radius: 3px;">
                            <div style="width: {pct}%; height: 100%; background: #00ffff;
                                       border-radius: 3px;"></div>
                        </div>
                    </div>
                </div>
            """

        html += f"""
            </div>

            <div style="margin-top: 15px; padding: 10px; background: #1a1a1a; border-radius: 5px;
                       text-align: center; font-size: 11px;">
                <div style="color: #888; margin-bottom: 5px;">ML Status:</div>
                <div style="color: #00ff00; font-weight: bold;">
                    {'✓ DQN Active' if self.agent else '✗ Random Mode'}
                </div>
                <div style="color: #888; margin-top: 5px;">
                    Total Predictions: {len(self.action_history)}
                </div>
            </div>
        </div>
        """
        return html


# Global instance
demo_instance = None


def create_demo():
    """Create smooth arcade demo"""
    global demo_instance

    custom_css = """
    body { background: #000; color: #fff; }
    .gradio-container { background: #000 !important; }
    .gr-button { font-family: 'Courier New', monospace !important; }
    """

    with gr.Blocks(title="Smooth Arcade - Multi-Game AI", css=custom_css) as demo:

        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #1a1a1a, #2a2a2a, #1a1a1a);
                   border-bottom: 2px solid #333;">
            <h1 style="color: #ffff00; font-family: 'Courier New', monospace; margin: 0;">
                🎮 ALVIN - Multi-Game AI Foundation Agent
            </h1>
            <p style="color: #00ffff; font-family: 'Courier New', monospace; margin: 10px 0 0 0;">
                Human vs DQN Agent • Zero-Shot Transfer Learning
            </p>
        </div>
        """)

        # Main layout: Stats | Game | AI Metrics
        with gr.Row():
            # Left: Game Stats
            with gr.Column(scale=1):
                stats_panel = gr.HTML(label="📊 Stats")

            # Center: Game
            with gr.Column(scale=2):
                with gr.Row():
                    game_type = gr.Radio(
                        choices=["snake", "pacman", "dungeon", "local_view"],
                        value="pacman",
                        label="🎮 Game"
                    )
                    difficulty = gr.Slider(0, 2, value=1, step=1, label="Difficulty")

                game_display = gr.Image(label="Game View", type="pil", height=550)

                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset", variant="secondary")
                    load_btn = gr.Button("📥 Load AI", variant="secondary")
                    model_path = gr.Textbox(value="checkpoints/multi_game_enhanced_*_policy.pth",
                                          label="Model", scale=2)

                with gr.Row():
                    up_btn = gr.Button("⬆️")
                    play_human_btn = gr.Button("▶️ Play Human", variant="primary")
                    play_ai_btn = gr.Button("▶️ Play AI", variant="primary")
                    play_both_btn = gr.Button("⚡ Play Both", variant="stop")

                with gr.Row():
                    left_btn = gr.Button("⬅️")
                    down_btn = gr.Button("⬇️")
                    right_btn = gr.Button("➡️")

            # Right: AI Metrics
            with gr.Column(scale=1):
                ai_metrics_panel = gr.HTML(label="🧠 AI Metrics")

        # Event handlers
        def load_model_wrapper(model_pattern):
            global demo_instance
            import glob
            files = glob.glob(model_pattern)
            if files:
                model_file = max(files, key=os.path.getctime)
                demo_instance = SmoothArcadeDemo(model_path=model_file)
                return demo_instance.load_model()
            demo_instance = SmoothArcadeDemo()
            return "✗ No model found"

        load_btn.click(load_model_wrapper, inputs=[model_path], outputs=[gr.Textbox(visible=False)])

        def reset_wrapper(game, diff):
            global demo_instance
            if not demo_instance:
                demo_instance = SmoothArcadeDemo()
            return demo_instance.reset_games(game, int(diff))

        reset_btn.click(reset_wrapper, inputs=[game_type, difficulty],
                       outputs=[game_display, stats_panel, ai_metrics_panel])

        # Human controls
        def human_step(action, game):
            if demo_instance and not demo_instance.human_stats['done'] and demo_instance.human_game:
                demo_instance.step_game(demo_instance.human_game, action)
                demo_instance.human_stats['steps'] += 1
                return (demo_instance.render_games(game), demo_instance.get_stats_html(game),
                       demo_instance.get_ai_metrics_html())
            return None, None, None

        up_btn.click(lambda g: human_step(0, g), inputs=[game_type],
                    outputs=[game_display, stats_panel, ai_metrics_panel])
        down_btn.click(lambda g: human_step(1, g), inputs=[game_type],
                      outputs=[game_display, stats_panel, ai_metrics_panel])
        left_btn.click(lambda g: human_step(2, g), inputs=[game_type],
                      outputs=[game_display, stats_panel, ai_metrics_panel])
        right_btn.click(lambda g: human_step(3, g), inputs=[game_type],
                       outputs=[game_display, stats_panel, ai_metrics_panel])

        # Auto-play
        def auto_play_both(game):
            global demo_instance
            if not demo_instance:
                demo_instance = SmoothArcadeDemo()

            img, stats, metrics = demo_instance.reset_games(game, 1)
            yield img, stats, metrics
            time.sleep(0.1)

            for step in range(500):
                if demo_instance.human_stats['done'] and demo_instance.ai_stats['done']:
                    break

                if not demo_instance.human_stats['done'] and demo_instance.human_game:
                    h_action = np.random.randint(4)
                    demo_instance.step_game(demo_instance.human_game, h_action)
                    demo_instance.human_stats['steps'] += 1

                if not demo_instance.ai_stats['done'] and demo_instance.ai_game:
                    a_action = demo_instance.get_ai_action(demo_instance.ai_game)
                    demo_instance.step_game(demo_instance.ai_game, a_action)
                    demo_instance.ai_stats['steps'] += 1

                if step % 2 == 0:
                    yield (demo_instance.render_games(game), demo_instance.get_stats_html(game),
                          demo_instance.get_ai_metrics_html())
                    time.sleep(0.03)

            demo_instance.human_stats['done'] = True
            demo_instance.ai_stats['done'] = True
            yield (demo_instance.render_games(game), demo_instance.get_stats_html(game),
                  demo_instance.get_ai_metrics_html())

        play_both_btn.click(auto_play_both, inputs=[game_type],
                           outputs=[game_display, stats_panel, ai_metrics_panel])

        # Initialize
        demo.load(reset_wrapper, inputs=[game_type, difficulty],
                 outputs=[game_display, stats_panel, ai_metrics_panel])

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)

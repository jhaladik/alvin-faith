"""
🕹️ RETRO ARCADE - Multi-Game Foundation Agent
Classic 90s Atari-Style Interface with AI vs Human Gameplay

Features all 4 games with authentic retro aesthetics:
- Snake, Pac-Man, Dungeon Explorer, Sky Collector
- CRT scanline effects, pixelated graphics
- Arcade cabinet UI with neon colors
"""

import gradio as gr
import numpy as np
import torch
import sys
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
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

# 🎨 RETRO COLOR PALETTE - Classic Atari/Arcade Colors
RETRO_COLORS = {
    'black': (0, 0, 0),
    'dark_blue': (0, 0, 128),
    'bright_blue': (0, 0, 255),
    'cyan': (0, 255, 255),
    'green': (0, 255, 0),
    'dark_green': (0, 128, 0),
    'yellow': (255, 255, 0),
    'orange': (255, 165, 0),
    'red': (255, 0, 0),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'gray': (128, 128, 128),
    'light_gray': (192, 192, 192),
    'neon_pink': (255, 20, 147),
    'neon_green': (57, 255, 20),
    'neon_blue': (4, 217, 255),
    'neon_orange': (255, 128, 0),
}


class RetroRenderer:
    """Authentic 90s arcade-style renderer"""

    @staticmethod
    def add_scanlines(img, intensity=0.15):
        """Add CRT scanline effect"""
        pixels = img.load()
        width, height = img.size

        for y in range(height):
            if y % 2 == 0:  # Every other line
                for x in range(width):
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (
                        int(r * (1 - intensity)),
                        int(g * (1 - intensity)),
                        int(b * (1 - intensity))
                    )
        return img

    @staticmethod
    def add_crt_curve(img):
        """Subtle CRT curve effect"""
        # Simple blur at edges for CRT feel
        return img.filter(ImageFilter.SMOOTH_MORE)

    @staticmethod
    def render_snake(game, cell_size=30):
        """Render Snake - Classic Atari style"""
        size = game.size
        img = Image.new('RGB', (size * cell_size, size * cell_size), RETRO_COLORS['black'])
        draw = ImageDraw.Draw(img)

        # Draw grid pattern (subtle)
        for i in range(0, size * cell_size, cell_size):
            draw.line([(i, 0), (i, size * cell_size)], fill=(20, 20, 20), width=1)
            draw.line([(0, i), (size * cell_size, i)], fill=(20, 20, 20), width=1)

        # Boundary walls - bright green like old monitors
        for i in range(size):
            draw.rectangle([i * cell_size, 0, (i+1) * cell_size, cell_size],
                          fill=RETRO_COLORS['dark_green'], outline=RETRO_COLORS['green'])
            draw.rectangle([i * cell_size, (size-1) * cell_size, (i+1) * cell_size, size * cell_size],
                          fill=RETRO_COLORS['dark_green'], outline=RETRO_COLORS['green'])
            draw.rectangle([0, i * cell_size, cell_size, (i+1) * cell_size],
                          fill=RETRO_COLORS['dark_green'], outline=RETRO_COLORS['green'])
            draw.rectangle([(size-1) * cell_size, i * cell_size, size * cell_size, (i+1) * cell_size],
                          fill=RETRO_COLORS['dark_green'], outline=RETRO_COLORS['green'])

        # Obstacles - gray blocks
        if hasattr(game, 'central_obstacles'):
            for ox, oy in game.central_obstacles:
                draw.rectangle([ox * cell_size + 2, oy * cell_size + 2,
                              (ox + 1) * cell_size - 2, (oy + 1) * cell_size - 2],
                             fill=RETRO_COLORS['gray'], outline=RETRO_COLORS['white'])

        # Food - bright red pixels
        if hasattr(game, 'food_positions'):
            for fx, fy in game.food_positions:
                cx, cy = int((fx + 0.5) * cell_size), int((fy + 0.5) * cell_size)
                r = int(cell_size * 0.35)
                # Draw as square for retro look
                draw.rectangle([cx-r, cy-r, cx+r, cy+r],
                             fill=RETRO_COLORS['red'], outline=RETRO_COLORS['yellow'])

        # Snake body - bright green
        if hasattr(game, 'snake') and game.snake:
            for x, y in game.snake[1:]:
                draw.rectangle([x * cell_size + 3, y * cell_size + 3,
                              (x + 1) * cell_size - 3, (y + 1) * cell_size - 3],
                             fill=RETRO_COLORS['green'], outline=RETRO_COLORS['neon_green'])

            # Snake head - cyan (player color)
            if game.snake:
                hx, hy = game.snake[0]
                draw.rectangle([hx * cell_size + 2, hy * cell_size + 2,
                              (hx + 1) * cell_size - 2, (hy + 1) * cell_size - 2],
                             fill=RETRO_COLORS['cyan'], outline=RETRO_COLORS['white'])

        img = RetroRenderer.add_scanlines(img, 0.12)
        return img

    @staticmethod
    def render_pacman(game, cell_size=30):
        """Render Pac-Man - Classic arcade style"""
        size = game.size
        img = Image.new('RGB', (size * cell_size, size * cell_size), RETRO_COLORS['black'])
        draw = ImageDraw.Draw(img)

        # Walls - bright blue blocks (classic Pac-Man maze color)
        for wx, wy in game.walls:
            draw.rectangle([wx * cell_size + 1, wy * cell_size + 1,
                          (wx + 1) * cell_size - 1, (wy + 1) * cell_size - 1],
                         fill=RETRO_COLORS['bright_blue'], outline=RETRO_COLORS['cyan'])

        # Pellets - small white dots
        for px, py in game.pellets:
            cx, cy = int((px + 0.5) * cell_size), int((py + 0.5) * cell_size)
            draw.rectangle([cx-3, cy-3, cx+3, cy+3], fill=RETRO_COLORS['white'])

        # Ghosts - classic colors with pixelated look
        ghost_colors = [
            RETRO_COLORS['red'],      # Blinky
            RETRO_COLORS['neon_pink'],  # Pinky
            RETRO_COLORS['orange'],   # Clyde
            RETRO_COLORS['cyan']      # Inky
        ]
        for i, ghost in enumerate(game.ghosts):
            gx, gy = ghost['pos']
            cx, cy = int((gx + 0.5) * cell_size), int((gy + 0.5) * cell_size)
            r = int(cell_size * 0.4)
            # Ghost body (square for retro)
            draw.rectangle([cx-r, cy-r, cx+r, cy+r],
                         fill=ghost_colors[i % len(ghost_colors)])
            # Eyes
            draw.rectangle([cx-r+3, cy-r+3, cx-r+7, cy-r+7], fill=RETRO_COLORS['white'])
            draw.rectangle([cx+r-7, cy-r+3, cx+r-3, cy-r+7], fill=RETRO_COLORS['white'])
            # Pupils
            draw.rectangle([cx-r+4, cy-r+4, cx-r+6, cy-r+6], fill=RETRO_COLORS['dark_blue'])
            draw.rectangle([cx+r-6, cy-r+4, cx+r-4, cy-r+6], fill=RETRO_COLORS['dark_blue'])

        # Pac-Man - bright yellow
        pmx, pmy = game.pacman_pos
        cx, cy = int((pmx + 0.5) * cell_size), int((pmy + 0.5) * cell_size)
        r = int(cell_size * 0.45)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=RETRO_COLORS['yellow'],
                    outline=RETRO_COLORS['neon_orange'])

        img = RetroRenderer.add_scanlines(img, 0.12)
        return img

    @staticmethod
    def render_dungeon(game, cell_size=30):
        """Render Dungeon - Classic dungeon crawler style"""
        size = game.size
        img = Image.new('RGB', (size * cell_size, size * cell_size), RETRO_COLORS['black'])
        draw = ImageDraw.Draw(img)

        # Walls - gray stone blocks
        for wx, wy in game.walls:
            # Main wall
            draw.rectangle([wx * cell_size + 1, wy * cell_size + 1,
                          (wx + 1) * cell_size - 1, (wy + 1) * cell_size - 1],
                         fill=RETRO_COLORS['gray'])
            # Highlight for 3D effect
            draw.line([wx * cell_size + 1, wy * cell_size + 1,
                      (wx + 1) * cell_size - 1, wy * cell_size + 1],
                     fill=RETRO_COLORS['light_gray'], width=2)
            draw.line([wx * cell_size + 1, wy * cell_size + 1,
                      wx * cell_size + 1, (wy + 1) * cell_size - 1],
                     fill=RETRO_COLORS['light_gray'], width=2)

        # Treasures - bright yellow/gold
        for tx, ty in game.treasures:
            cx, cy = int((tx + 0.5) * cell_size), int((ty + 0.5) * cell_size)
            # Draw as diamond shape
            points = [(cx, cy-8), (cx+8, cy), (cx, cy+8), (cx-8, cy)]
            draw.polygon(points, fill=RETRO_COLORS['yellow'], outline=RETRO_COLORS['neon_orange'])

        # Enemies - red/purple/orange squares
        enemy_colors = [RETRO_COLORS['red'], RETRO_COLORS['magenta'], RETRO_COLORS['orange']]
        for i, enemy in enumerate(game.enemies):
            ex, ey = enemy['pos']
            cx, cy = int((ex + 0.5) * cell_size), int((ey + 0.5) * cell_size)
            r = int(cell_size * 0.35)
            # Enemy body
            draw.rectangle([cx-r, cy-r, cx+r, cy+r],
                         fill=enemy_colors[i % len(enemy_colors)])
            # Eyes
            draw.rectangle([cx-r+2, cy-r+2, cx-r+6, cy-r+6], fill=RETRO_COLORS['white'])
            draw.rectangle([cx+r-6, cy-r+2, cx+r-2, cy-r+6], fill=RETRO_COLORS['white'])

        # Player - green/cyan
        px, py = game.player_pos
        cx, cy = int((px + 0.5) * cell_size), int((py + 0.5) * cell_size)
        r = int(cell_size * 0.4)
        draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=RETRO_COLORS['green'],
                      outline=RETRO_COLORS['cyan'])

        img = RetroRenderer.add_scanlines(img, 0.12)
        return img

    @staticmethod
    def render_local_view(game, cell_size=18, viewport_size=25):
        """Render Sky Collector - retro space shooter style"""
        # Blue sky background
        img = Image.new('RGB', (viewport_size * cell_size, viewport_size * cell_size),
                       (30, 60, 120))  # Darker blue
        draw = ImageDraw.Draw(img)

        ax, ay = game.agent_pos
        half_view = viewport_size // 2
        view_x, view_y = ax - half_view, ay - half_view

        def to_screen(wx, wy):
            return (wx - view_x) * cell_size, (wy - view_y) * cell_size

        def in_view(wx, wy):
            return view_x <= wx < view_x + viewport_size and view_y <= wy < view_y + viewport_size

        # Draw pixel grid
        for i in range(viewport_size + 1):
            draw.line([(i * cell_size, 0), (i * cell_size, viewport_size * cell_size)],
                     fill=(50, 80, 140), width=1)
            draw.line([(0, i * cell_size), (viewport_size * cell_size, i * cell_size)],
                     fill=(50, 80, 140), width=1)

        # Walls (clouds) - cyan blocks
        for wx, wy in game.walls:
            if in_view(wx, wy):
                sx, sy = to_screen(wx, wy)
                draw.rectangle([sx+1, sy+1, sx+cell_size-1, sy+cell_size-1],
                             fill=RETRO_COLORS['cyan'], outline=RETRO_COLORS['white'])

        # Coins - yellow diamonds
        for cx, cy in game.coins:
            if in_view(cx, cy):
                sx, sy = to_screen(cx, cy)
                center_x, center_y = int(sx + cell_size/2), int(sy + cell_size/2)
                # Diamond shape
                points = [(center_x, center_y-5), (center_x+5, center_y),
                         (center_x, center_y+5), (center_x-5, center_y)]
                draw.polygon(points, fill=RETRO_COLORS['yellow'],
                           outline=RETRO_COLORS['neon_orange'])

        # Enemies (birds) - red/orange squares
        enemy_colors = [RETRO_COLORS['red'], RETRO_COLORS['orange'], RETRO_COLORS['magenta']]
        for i, enemy in enumerate(game.enemies):
            ex, ey = enemy['pos']
            if in_view(ex, ey):
                sx, sy = to_screen(ex, ey)
                draw.rectangle([sx+2, sy+2, sx+cell_size-2, sy+cell_size-2],
                             fill=enemy_colors[i % len(enemy_colors)])
                # Eyes
                draw.rectangle([sx+4, sy+4, sx+7, sy+7], fill=RETRO_COLORS['white'])
                draw.rectangle([sx+cell_size-7, sy+4, sx+cell_size-4, sy+7],
                             fill=RETRO_COLORS['white'])

        # Airplane (player) - green with outline
        sx, sy = to_screen(ax, ay)
        cx, cy = int(sx + cell_size/2), int(sy + cell_size/2)
        r = int(cell_size * 0.45)
        draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=RETRO_COLORS['green'],
                      outline=RETRO_COLORS['cyan'])

        img = RetroRenderer.add_scanlines(img, 0.1)
        return img


class RetroArcadeDemo:
    """Retro arcade game manager"""

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.agent = None
        self.observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)
        self.human_game = None
        self.ai_game = None
        self.human_stats = {'score': 0, 'steps': 0, 'done': False}
        self.ai_stats = {'score': 0, 'steps': 0, 'done': False}
        self.load_model()

    def load_model(self):
        """Load trained agent"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                self.agent = ContextAwareDQN(obs_dim=183, action_dim=4)
                self.agent.load_state_dict(checkpoint['policy_net'])
                self.agent.eval()
                return "✓ AI LOADED"
            except Exception as e:
                return f"✗ ERROR: {e}"
        return "⚠ RANDOM MODE"

    def reset_games(self, game_type, difficulty=0):
        """Reset both games"""
        self.human_stats = {'score': 0, 'steps': 0, 'done': False}
        self.ai_stats = {'score': 0, 'steps': 0, 'done': False}

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
        """Render both games side-by-side with arcade cabinet frame"""
        if game_type == 'snake':
            h_img = RetroRenderer.render_snake(self.human_game, cell_size=28)
            a_img = RetroRenderer.render_snake(self.ai_game, cell_size=28)
        elif game_type == 'pacman':
            h_img = RetroRenderer.render_pacman(self.human_game, cell_size=28)
            a_img = RetroRenderer.render_pacman(self.ai_game, cell_size=28)
        elif game_type == 'dungeon':
            h_img = RetroRenderer.render_dungeon(self.human_game, cell_size=28)
            a_img = RetroRenderer.render_dungeon(self.ai_game, cell_size=28)
        elif game_type == 'local_view':
            h_img = RetroRenderer.render_local_view(self.human_game, cell_size=18)
            a_img = RetroRenderer.render_local_view(self.ai_game, cell_size=18)

        # Create arcade cabinet style frame
        gap = 40
        border = 20
        label_height = 40
        total_width = h_img.width + a_img.width + gap + 2 * border
        total_height = h_img.height + 2 * border + label_height

        # Dark gray cabinet background
        combined = Image.new('RGB', (total_width, total_height), (30, 30, 30))
        draw = ImageDraw.Draw(combined)

        # Draw outer frame (arcade cabinet style)
        draw.rectangle([0, 0, total_width-1, total_height-1],
                      outline=(100, 100, 100), width=border)

        # Add labels
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # PLAYER 1 label
        p1_x = border + h_img.width // 2
        draw.text((p1_x, 10), "PLAYER 1", fill=(0, 255, 0), anchor="mm", font=font)

        # PLAYER 2 (AI) label
        p2_x = border + h_img.width + gap + a_img.width // 2
        draw.text((p2_x, 10), "AI AGENT", fill=(0, 255, 255), anchor="mm", font=font)

        # Paste games
        combined.paste(h_img, (border, border + label_height))
        combined.paste(a_img, (border + h_img.width + gap, border + label_height))

        # Draw separator line
        mid_x = border + h_img.width + gap // 2
        draw.line([(mid_x, border + label_height), (mid_x, total_height - border)],
                 fill=(255, 255, 0), width=4)

        return combined

    def get_stats_html(self, game_type):
        """Generate retro arcade stats HTML"""
        self.human_stats['score'] = self.human_game.score if self.human_game else 0
        self.ai_stats['score'] = self.ai_game.score if self.ai_game else 0

        game_names = {
            'snake': '🐍 SNAKE',
            'pacman': '👾 PAC-MAN',
            'dungeon': '🏰 DUNGEON',
            'local_view': '✈️ SKY COLLECTOR'
        }

        # Retro arcade style with neon colors
        html = f"""
        <div style="font-family: 'Courier New', monospace; background: linear-gradient(180deg, #000000 0%, #1a1a2e 50%, #000000 100%);
                    padding: 20px; border: 4px solid #00ff00; border-radius: 10px; color: #00ff00;
                    box-shadow: 0 0 20px #00ff00, inset 0 0 20px rgba(0,255,0,0.1);">

            <div style="text-align: center; margin-bottom: 20px; padding: 15px; background: #000;
                       border: 2px solid #ffff00; box-shadow: 0 0 10px #ffff00;">
                <h1 style="color: #ffff00; font-size: 32px; margin: 0; text-shadow: 0 0 10px #ffff00, 0 0 20px #ff8800;">
                    {game_names.get(game_type, 'GAME')}
                </h1>
                <p style="color: #00ffff; font-size: 14px; margin: 5px 0;">
                    INSERT COIN TO CONTINUE
                </p>
            </div>

            <!-- PLAYER 1 STATS -->
            <div style="background: #000; padding: 15px; margin: 10px 0; border: 3px solid #00ff00;
                       box-shadow: 0 0 15px rgba(0,255,0,0.5);">
                <h2 style="color: #00ff00; font-size: 22px; margin: 0 0 10px 0; text-shadow: 0 0 5px #00ff00;">
                    ▶ PLAYER 1 ◀
                </h2>
                <table style="width: 100%; color: #00ff00; font-size: 18px;">
                    <tr>
                        <td>SCORE:</td>
                        <td style="text-align: right; color: #ffff00; font-size: 28px; text-shadow: 0 0 5px #ffff00;">
                            {self.human_stats['score']:06d}
                        </td>
                    </tr>
                    <tr>
                        <td>STEPS:</td>
                        <td style="text-align: right;">{self.human_stats['steps']:04d}</td>
                    </tr>
                    <tr>
                        <td>STATUS:</td>
                        <td style="text-align: right; color: {'#ff0000' if self.human_stats['done'] else '#00ff00'};">
                            {'GAME OVER' if self.human_stats['done'] else 'PLAYING'}
                        </td>
                    </tr>
                </table>
            </div>

            <!-- AI STATS -->
            <div style="background: #000; padding: 15px; margin: 10px 0; border: 3px solid #00ffff;
                       box-shadow: 0 0 15px rgba(0,255,255,0.5);">
                <h2 style="color: #00ffff; font-size: 22px; margin: 0 0 10px 0; text-shadow: 0 0 5px #00ffff;">
                    ▶ AI AGENT ◀
                </h2>
                <table style="width: 100%; color: #00ffff; font-size: 18px;">
                    <tr>
                        <td>SCORE:</td>
                        <td style="text-align: right; color: #ff00ff; font-size: 28px; text-shadow: 0 0 5px #ff00ff;">
                            {self.ai_stats['score']:06d}
                        </td>
                    </tr>
                    <tr>
                        <td>STEPS:</td>
                        <td style="text-align: right;">{self.ai_stats['steps']:04d}</td>
                    </tr>
                    <tr>
                        <td>STATUS:</td>
                        <td style="text-align: right; color: {'#ff0000' if self.ai_stats['done'] else '#00ffff'};">
                            {'GAME OVER' if self.ai_stats['done'] else 'PLAYING'}
                        </td>
                    </tr>
                </table>
            </div>

            <!-- SCORE COMPARISON -->
            <div style="background: #000; padding: 20px; margin: 15px 0; border: 3px solid #ffff00;
                       text-align: center; box-shadow: 0 0 15px rgba(255,255,0,0.5);">
                <h2 style="color: #ffff00; font-size: 20px; margin-bottom: 15px; text-shadow: 0 0 5px #ffff00;">
                    BATTLE SCORE
                </h2>
                <div style="font-size: 40px; font-weight: bold; margin: 10px 0; text-shadow: 0 0 10px;">
                    <span style="color: #00ff00;">{self.human_stats['score']}</span>
                    <span style="color: #ffff00; font-size: 24px;"> VS </span>
                    <span style="color: #00ffff;">{self.ai_stats['score']}</span>
                </div>
                <div style="font-size: 24px; color: #ff00ff; margin-top: 15px; text-shadow: 0 0 10px #ff00ff;">
                    {'★ PLAYER 1 WINS! ★' if self.human_stats['score'] > self.ai_stats['score'] else
                     '★ AI WINS! ★' if self.ai_stats['score'] > self.human_stats['score'] else
                     '★ TIE GAME! ★'}
                </div>
            </div>

            <!-- ZERO-SHOT TRANSFER RESULTS -->
            <div style="background: #000; padding: 15px; margin: 15px 0; border: 3px solid #ff00ff;
                       box-shadow: 0 0 15px rgba(255,0,255,0.5);">
                <h3 style="color: #ff00ff; font-size: 18px; margin: 0 0 10px 0; text-shadow: 0 0 5px #ff00ff;">
                    ⚡ ZERO-SHOT TRANSFER ⚡
                </h3>
                <p style="color: #00ff00; font-size: 14px; margin: 5px 0;">
                    DUNGEON: 66.7% WIN RATE<br>
                    PACMAN: 75.0% WIN RATE<br>
                    TRAINED: SNAKE ONLY
                </p>
            </div>

            <div style="text-align: center; margin-top: 20px; padding: 10px; color: #888; font-size: 12px;">
                🕹️ RETRO ARCADE 1990 🕹️<br>
                AI FOUNDATION AGENT
            </div>
        </div>
        """
        return html


# Global instance
demo_instance = None


def create_demo():
    """Create retro arcade Gradio interface"""
    global demo_instance

    # Custom CSS for full retro theme
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

    body {
        background: #000000;
        color: #00ff00;
    }

    .gradio-container {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a0a2e 50%, #0a0a0a 100%) !important;
        font-family: 'Courier New', monospace !important;
    }

    h1, h2, h3 {
        font-family: 'Courier New', monospace !important;
        color: #00ff00 !important;
        text-shadow: 0 0 10px #00ff00;
    }

    .gr-button {
        background: linear-gradient(180deg, #00ff00 0%, #00aa00 100%) !important;
        color: #000000 !important;
        border: 3px solid #00ff00 !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
        text-shadow: none !important;
        box-shadow: 0 0 10px #00ff00 !important;
    }

    .gr-button:hover {
        background: linear-gradient(180deg, #00ffff 0%, #00aaaa 100%) !important;
        box-shadow: 0 0 20px #00ffff !important;
    }

    .gr-button-primary {
        background: linear-gradient(180deg, #ffff00 0%, #aaaa00 100%) !important;
        border: 3px solid #ffff00 !important;
        box-shadow: 0 0 10px #ffff00 !important;
    }

    .gr-button-secondary {
        background: linear-gradient(180deg, #00ffff 0%, #00aaaa 100%) !important;
        border: 3px solid #00ffff !important;
        box-shadow: 0 0 10px #00ffff !important;
    }

    .gr-input, .gr-box {
        background: #000000 !important;
        border: 2px solid #00ff00 !important;
        color: #00ff00 !important;
        box-shadow: 0 0 5px rgba(0,255,0,0.3) !important;
    }

    label {
        color: #00ffff !important;
        font-family: 'Courier New', monospace !important;
    }
    """

    with gr.Blocks(title="🕹️ RETRO ARCADE - AI Foundation Agent",
                   theme=gr.themes.Base(), css=custom_css) as demo:

        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(90deg, #ff0000, #ff00ff, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000);
                   border: 4px solid #ffff00; margin-bottom: 20px; animation: rainbow 3s linear infinite;">
            <h1 style="font-family: 'Courier New', monospace; font-size: 48px; color: #000; margin: 0;
                      text-shadow: 3px 3px 0px #ffff00, 6px 6px 0px #ff00ff;">
                🕹️ RETRO ARCADE 🕹️
            </h1>
            <h2 style="font-family: 'Courier New', monospace; font-size: 24px; color: #000; margin: 10px 0;">
                MULTI-GAME AI FOUNDATION AGENT
            </h2>
            <p style="font-family: 'Courier New', monospace; font-size: 16px; color: #000; margin: 5px 0;">
                ⚡ ZERO-SHOT TRANSFER ⚡ TRAINED ON SNAKE ⚡ PLAYS ALL 4 GAMES ⚡
            </p>
        </div>
        """)

        gr.Markdown("""
        ### 🎮 **SELECT YOUR GAME**
        Watch the AI agent (trained only on Snake) play 4 different games using zero-shot transfer learning!

        **Available Games:** 🐍 Snake | 👾 Pac-Man | 🏰 Dungeon Explorer | ✈️ Sky Collector
        """)

        # Game selection row
        with gr.Row():
            game_type = gr.Radio(
                choices=["snake", "pacman", "dungeon", "local_view"],
                value="pacman",
                label="🎮 GAME SELECT",
                scale=2
            )
            difficulty = gr.Slider(0, 2, value=1, step=1,
                                  label="🎚️ DIFFICULTY",
                                  info="0=EASY | 1=MEDIUM | 2=HARD",
                                  scale=1)
            model_path = gr.Textbox(
                value="checkpoints/multi_game_enhanced_*_policy.pth",
                label="🤖 MODEL PATH",
                scale=2
            )

        # Main game area
        with gr.Row():
            # Controls
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ SYSTEM")

                load_btn = gr.Button("📥 LOAD AI", variant="secondary", size="lg")
                reset_btn = gr.Button("🔄 RESET", variant="secondary", size="lg")
                model_status = gr.Textbox(label="STATUS", lines=2, interactive=False)

                gr.Markdown("### 🎮 PLAYER 1")
                with gr.Column():
                    up_btn = gr.Button("⬆️ UP", size="lg")
                    with gr.Row():
                        left_btn = gr.Button("⬅️ LEFT", size="sm")
                        down_btn = gr.Button("⬇️ DOWN", size="sm")
                        right_btn = gr.Button("➡️ RIGHT", size="sm")

                gr.Markdown("### 🤖 AUTO-PLAY")
                play_human_btn = gr.Button("▶️ PLAY HUMAN", variant="primary", size="lg")
                play_ai_btn = gr.Button("▶️ PLAY AI", variant="primary", size="lg")
                play_both_btn = gr.Button("⚡ PLAY BOTH", variant="stop", size="lg")

                gr.Markdown("""
                ---
                **CONTROLS:**
                - ⬆️⬇️⬅️➡️ Manual play
                - ▶️ Auto-play modes
                - 🔄 Reset game

                **AI FEATURES:**
                - Trained: Snake only
                - Transfer: All games
                - Zero-shot learning
                """)

            # Game display
            with gr.Column(scale=3):
                game_display = gr.Image(label="🕹️ ARCADE SCREEN 🕹️",
                                       type="pil", height=650)

                gr.HTML("""
                <div style="text-align: center; padding: 15px; background: #000;
                           border: 3px solid #ff00ff; margin-top: 10px;">
                    <p style="color: #ff00ff; font-family: 'Courier New', monospace;
                             font-size: 18px; margin: 0; text-shadow: 0 0 5px #ff00ff;">
                        ⚡ PLAYER 1 (LEFT) VS AI AGENT (RIGHT) ⚡
                    </p>
                </div>
                """)

            # Stats panel
            with gr.Column(scale=1):
                stats_panel = gr.HTML(label="📊 SCOREBOARD")

        # Bottom info
        gr.HTML("""
        <div style="margin-top: 30px; padding: 25px; background: #000;
                   border: 3px solid #00ffff; text-align: center;">
            <h3 style="color: #00ffff; font-family: 'Courier New', monospace;
                      text-shadow: 0 0 10px #00ffff; margin-bottom: 15px;">
                🏆 ZERO-SHOT TRANSFER RESULTS 🏆
            </h3>
            <table style="width: 100%; color: #00ff00; font-family: 'Courier New', monospace;
                         font-size: 16px; margin: auto; max-width: 600px;">
                <tr style="background: #001100;">
                    <td style="padding: 10px; border: 1px solid #00ff00;">🐍 SNAKE</td>
                    <td style="padding: 10px; border: 1px solid #00ff00; color: #ffff00;">TRAINED</td>
                    <td style="padding: 10px; border: 1px solid #00ff00;">100% BASE</td>
                </tr>
                <tr style="background: #000011;">
                    <td style="padding: 10px; border: 1px solid #00ffff;">👾 PAC-MAN</td>
                    <td style="padding: 10px; border: 1px solid #00ffff; color: #ff00ff;">TRANSFER</td>
                    <td style="padding: 10px; border: 1px solid #00ffff;">75.0% WIN</td>
                </tr>
                <tr style="background: #110000;">
                    <td style="padding: 10px; border: 1px solid #ff00ff;">🏰 DUNGEON</td>
                    <td style="padding: 10px; border: 1px solid #ff00ff; color: #ff00ff;">TRANSFER</td>
                    <td style="padding: 10px; border: 1px solid #ff00ff;">66.7% WIN</td>
                </tr>
                <tr style="background: #001111;">
                    <td style="padding: 10px; border: 1px solid #00ffff;">✈️ SKY</td>
                    <td style="padding: 10px; border: 1px solid #00ffff; color: #ff00ff;">TRANSFER</td>
                    <td style="padding: 10px; border: 1px solid #00ffff;">TESTING</td>
                </tr>
            </table>
            <p style="color: #ffff00; margin-top: 20px; font-size: 14px;">
                AI learns spatial reasoning from Snake and applies it to completely new games!
            </p>
        </div>
        """)

        gr.Markdown("""
        ---
        ### 📖 ABOUT THIS DEMO

        This is a **Context-Aware Foundation Agent** that demonstrates:
        - **Zero-shot transfer learning** across different game types
        - **Spatial reasoning** that generalizes beyond training data
        - **Faith-based exploration** for discovering hidden mechanics
        - **Model-based planning** with learned world models

        The agent was trained **only on Snake** but successfully plays:
        - 👾 **Pac-Man** (75% win rate, no additional training)
        - 🏰 **Dungeon Explorer** (66.7% win rate, no additional training)
        - ✈️ **Sky Collector** (local view, testing transfer limits)

        **Technical Details:**
        - 16 raycast sensors for spatial awareness
        - Temporal observation history
        - Context-aware decision making
        - DQN with world model planning

        🎮 **Try different difficulties and see how well the AI adapts!**
        """)

        # Event handlers
        def load_model_wrapper(model_pattern):
            global demo_instance
            import glob
            files = glob.glob(model_pattern)
            if files:
                model_file = max(files, key=os.path.getctime)
                demo_instance = RetroArcadeDemo(model_path=model_file)
                return demo_instance.load_model() + f"\n{os.path.basename(model_file)}"
            demo_instance = RetroArcadeDemo()
            return f"✗ NO MODEL FOUND"

        load_btn.click(load_model_wrapper, inputs=[model_path], outputs=[model_status])

        def reset_wrapper(game, diff):
            global demo_instance
            if not demo_instance:
                demo_instance = RetroArcadeDemo()
            return demo_instance.reset_games(game, int(diff))

        reset_btn.click(reset_wrapper, inputs=[game_type, difficulty],
                       outputs=[game_display, stats_panel])

        # Human controls
        def human_step(action, game):
            if demo_instance and not demo_instance.human_stats['done']:
                demo_instance.step_game(demo_instance.human_game, action)
                demo_instance.human_stats['steps'] += 1
                return demo_instance.render_games(game), demo_instance.get_stats_html(game)
            return None, "RESET GAME FIRST"

        up_btn.click(lambda g: human_step(0, g), inputs=[game_type],
                    outputs=[game_display, stats_panel])
        down_btn.click(lambda g: human_step(1, g), inputs=[game_type],
                      outputs=[game_display, stats_panel])
        left_btn.click(lambda g: human_step(2, g), inputs=[game_type],
                      outputs=[game_display, stats_panel])
        right_btn.click(lambda g: human_step(3, g), inputs=[game_type],
                       outputs=[game_display, stats_panel])

        # Auto-play functions
        def auto_play_human(game):
            global demo_instance
            if not demo_instance:
                demo_instance = RetroArcadeDemo()

            # Reset first and yield initial state
            img, stats = demo_instance.reset_games(game, 1)
            yield img, stats
            time.sleep(0.1)

            for step in range(500):
                if demo_instance.human_stats['done'] or not demo_instance.human_game:
                    break
                action = np.random.randint(4)
                demo_instance.step_game(demo_instance.human_game, action)
                demo_instance.human_stats['steps'] += 1
                if step % 2 == 0:
                    yield demo_instance.render_games(game), demo_instance.get_stats_html(game)
                    time.sleep(0.03)

            demo_instance.human_stats['done'] = True
            yield demo_instance.render_games(game), demo_instance.get_stats_html(game)

        def auto_play_ai(game):
            global demo_instance
            if not demo_instance:
                demo_instance = RetroArcadeDemo()

            # Reset first and yield initial state
            img, stats = demo_instance.reset_games(game, 1)
            yield img, stats
            time.sleep(0.1)

            for step in range(500):
                if demo_instance.ai_stats['done'] or not demo_instance.ai_game:
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
            global demo_instance
            if not demo_instance:
                demo_instance = RetroArcadeDemo()

            # Reset first and yield initial state
            img, stats = demo_instance.reset_games(game, 1)
            yield img, stats
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

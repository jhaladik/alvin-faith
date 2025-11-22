"""
Web-based Game Streamer - No VNC needed!
Streams pygame frames directly to browser via MJPEG
"""
import pygame
import torch
import numpy as np
import io
import sys
import os
from threading import Thread, Lock
from flask import Flask, Response, render_template_string, request, jsonify
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.enhanced_snake_game import EnhancedSnakeGame
from src.core.simple_dungeon_game import SimpleDungeonGame
from src.core.local_view_game import LocalViewGame
from src.core.simple_pacman_game import SimplePacManGame
from src.context_aware_agent import ContextAwareDQN, add_context_to_observation
from src.core.expanded_temporal_observer import ExpandedTemporalObserver

# Flask app
app = Flask(__name__)

# Global state
game_state = {
    'current_game': 0,
    'difficulty': 3,
    'paused': False,
    'frame': None,
    'lock': Lock(),
    'running': True,
    'episodes': 0,
    'victories': 0,
    'deaths': 0,
    'human_playing': False,  # Toggle between AI and Human
    'manual_action': None,   # Store manual input action
}

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Faith Multi-Game Agent</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: #1a1a1a;
            color: white;
            font-family: Arial, sans-serif;
            overflow-x: hidden;
        }
        .container {
            display: flex;
            min-height: 100vh;
        }
        .sidebar {
            width: 300px;
            background: #2a2a2a;
            padding: 20px;
            box-shadow: 2px 0 10px rgba(0,0,0,0.5);
            display: flex;
            flex-direction: column;
        }
        .main-content {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        #game-frame {
            width: 920px;
            height: 600px;
            border: 3px solid #00ff00;
            border-radius: 8px;
            image-rendering: pixelated;
            image-rendering: crisp-edges;
            box-shadow: 0 0 20px rgba(0,255,0,0.3);
        }
        h1 {
            color: #00ff00;
            margin: 0 0 20px 0;
            font-size: 24px;
            text-align: center;
        }
        h2 {
            color: #00aaff;
            font-size: 18px;
            margin: 25px 0 15px 0;
            border-bottom: 2px solid #00aaff;
            padding-bottom: 8px;
        }
        .game-button {
            background: #00aa00;
            color: white;
            border: none;
            padding: 15px;
            margin: 8px 0;
            border-radius: 8px;
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
            width: 100%;
            transition: all 0.3s;
        }
        .game-button:hover {
            background: #00ff00;
            transform: translateX(5px);
        }
        .game-button.active {
            background: #0066ff;
            box-shadow: 0 0 15px rgba(0,102,255,0.5);
        }
        .control-group {
            margin: 15px 0;
        }
        .difficulty-control {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #333;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .diff-btn {
            background: #555;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 20px;
            font-weight: bold;
        }
        .diff-btn:hover {
            background: #777;
        }
        .diff-value {
            font-size: 24px;
            font-weight: bold;
            color: #00ff00;
        }
        .action-button {
            background: #444;
            color: white;
            border: none;
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            transition: all 0.3s;
        }
        .action-button:hover {
            background: #666;
        }
        .mode-toggle {
            background: linear-gradient(135deg, #00aa00 0%, #00dd00 100%);
            font-size: 18px;
            font-weight: bold;
        }
        .mode-toggle:hover {
            background: linear-gradient(135deg, #00dd00 0%, #00ff00 100%);
        }
        .mode-toggle.human {
            background: linear-gradient(135deg, #ff6600 0%, #ff8800 100%);
        }
        .mode-toggle.human:hover {
            background: linear-gradient(135deg, #ff8800 0%, #ffaa00 100%);
        }
        .mode-hint {
            text-align: center;
            padding: 8px;
            background: #333;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 13px;
            color: #aaa;
        }
        .stats-section {
            margin-top: auto;
            padding-top: 20px;
            border-top: 2px solid #444;
        }
        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            font-size: 16px;
        }
        .stat-label {
            color: #aaa;
        }
        .stat-value {
            color: #00ff00;
            font-weight: bold;
        }
        .current-game {
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, #0066ff 0%, #0099ff 100%);
            border-radius: 8px;
            margin: 10px 0 20px 0;
            font-size: 20px;
            font-weight: bold;
            color: white;
            box-shadow: 0 4px 15px rgba(0,102,255,0.4);
            border: 2px solid #00aaff;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Sidebar with controls -->
        <div class="sidebar">
            <h1>🎮 Faith Agent</h1>

            <div class="current-game">
                <span id="current-game">Snake</span>
            </div>

            <h2>Select Game</h2>
            <button class="game-button active" onclick="switchGame(0)" id="game0">🐍 Snake</button>
            <button class="game-button" onclick="switchGame(1)" id="game1">🏰 Dungeon</button>
            <button class="game-button" onclick="switchGame(2)" id="game2">👁️ Local View</button>
            <button class="game-button" onclick="switchGame(3)" id="game3">👻 PacMan</button>

            <h2>Difficulty</h2>
            <div class="difficulty-control">
                <button class="diff-btn" onclick="changeDifficulty(-1)">−</button>
                <span class="diff-value" id="difficulty">3</span>
                <button class="diff-btn" onclick="changeDifficulty(1)">+</button>
            </div>

            <h2>Controls</h2>
            <button class="action-button mode-toggle" onclick="toggleMode()" id="mode-btn">
                🤖 AI Playing
            </button>
            <div class="mode-hint" id="mode-hint">
                AI is playing automatically
            </div>

            <h2>Actions</h2>
            <button class="action-button" onclick="togglePause()">⏸️ Pause / Resume</button>
            <button class="action-button" onclick="reset()">↻ Reset Episode</button>

            <div class="stats-section">
                <h2>📊 Statistics</h2>
                <div class="stat-item">
                    <span class="stat-label">Episodes:</span>
                    <span class="stat-value" id="episodes">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Victories:</span>
                    <span class="stat-value" id="victories">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Deaths:</span>
                    <span class="stat-value" id="deaths">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Win Rate:</span>
                    <span class="stat-value" id="win-rate">0%</span>
                </div>
            </div>
        </div>

        <!-- Main game display -->
        <div class="main-content">
            <img id="game-frame" src="/video_feed" alt="Game Stream">
        </div>
    </div>

    <script>
        const gameNames = ['Snake', 'Dungeon', 'Local View', 'PacMan'];
        let currentGame = 0;

        function switchGame(gameId) {
            fetch('/control?action=switch_game&game=' + gameId);
            currentGame = gameId;
            updateUI();
        }

        function changeDifficulty(delta) {
            fetch('/control?action=difficulty&delta=' + delta);
        }

        function togglePause() {
            fetch('/control?action=pause');
        }

        function reset() {
            fetch('/control?action=reset');
        }

        function toggleMode() {
            fetch('/control?action=toggle_mode')
                .then(r => r.json())
                .then(data => {
                    const btn = document.getElementById('mode-btn');
                    const hint = document.getElementById('mode-hint');
                    if (data.human_playing) {
                        btn.className = 'action-button mode-toggle human';
                        btn.innerHTML = '👤 Human Playing';
                        hint.textContent = 'Use arrow keys: ↑↓←→ to play';
                    } else {
                        btn.className = 'action-button mode-toggle';
                        btn.innerHTML = '🤖 AI Playing';
                        hint.textContent = 'AI is playing automatically';
                    }
                });
        }

        function sendManualAction(action) {
            fetch('/control?action=manual&direction=' + action);
        }

        function updateUI() {
            // Update active game button
            for (let i = 0; i < 4; i++) {
                document.getElementById('game' + i).className = (i === currentGame) ? 'active' : '';
            }
            document.getElementById('current-game').textContent = gameNames[currentGame];
        }

        // Poll for stats updates
        setInterval(() => {
            fetch('/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('difficulty').textContent = data.difficulty;
                    document.getElementById('episodes').textContent = data.episodes;
                    document.getElementById('victories').textContent = data.victories;
                    document.getElementById('deaths').textContent = data.deaths;

                    // Calculate and update win rate
                    if (data.episodes > 0) {
                        const winRate = ((data.victories / data.episodes) * 100).toFixed(1);
                        document.getElementById('win-rate').textContent = winRate + '%';
                    } else {
                        document.getElementById('win-rate').textContent = '0%';
                    }

                    currentGame = data.current_game;
                    updateUI();
                });
        }, 1000);

        // Keyboard support
        document.addEventListener('keydown', (e) => {
            // Check if in human mode for arrow keys
            fetch('/stats')
                .then(r => r.json())
                .then(data => {
                    if (data.human_playing) {
                        // In human mode - arrow keys control the game
                        if (e.key === 'ArrowUp') {
                            e.preventDefault();
                            sendManualAction(0); // Up
                        } else if (e.key === 'ArrowDown') {
                            e.preventDefault();
                            sendManualAction(1); // Down
                        } else if (e.key === 'ArrowLeft') {
                            e.preventDefault();
                            sendManualAction(2); // Left
                        } else if (e.key === 'ArrowRight') {
                            e.preventDefault();
                            sendManualAction(3); // Right
                        }
                    } else {
                        // In AI mode - arrow keys change difficulty
                        if (e.key === 'ArrowUp') {
                            changeDifficulty(1);
                        } else if (e.key === 'ArrowDown') {
                            changeDifficulty(-1);
                        }
                    }
                });

            // Number keys, space, and R always work
            if (e.key >= '1' && e.key <= '4') {
                switchGame(parseInt(e.key) - 1);
            } else if (e.key === ' ') {
                e.preventDefault();
                togglePause();
            } else if (e.key === 'r' || e.key === 'R') {
                reset();
            } else if (e.key === 'm' || e.key === 'M') {
                toggleMode();
            }
        });

        updateUI();
    </script>
</body>
</html>
"""


class HeadlessGameRenderer:
    """Runs games headless and renders to surface"""

    def __init__(self, model_path, width=920, height=600):
        self.width = width
        self.height = height

        # Initialize pygame in headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        self.screen = pygame.Surface((width, height))

        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.agent = ContextAwareDQN(obs_dim=183, action_dim=4)
        self.agent.load_state_dict(checkpoint['policy_net'])
        self.agent.eval()

        # Observer
        self.observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)

        # Game state
        self.game_names = ["Snake", "Dungeon", "Local View", "PacMan"]
        self.create_game()

        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 32)

        print(f"Headless renderer initialized: {width}x{height}")

    def create_game(self):
        """Create the selected game"""
        game_idx = game_state['current_game']
        level = game_state['difficulty']

        if game_idx == 0:  # Snake
            self.game = EnhancedSnakeGame(
                size=20, initial_pellets=7, max_pellets=12,
                food_timeout=150, obstacle_level=level,
                max_steps=400, max_total_food=40
            )
        elif game_idx == 1:  # Dungeon
            self.game = SimpleDungeonGame(
                size=20, num_treasures=3, enemy_level=level, max_steps=500
            )
        elif game_idx == 2:  # Local View
            self.game = LocalViewGame(
                world_size=40, num_coins=20, enemy_level=level,
                max_steps=800, viewport_size=25
            )
        elif game_idx == 3:  # PacMan
            self.game = SimplePacManGame(
                size=20, num_pellets=30, ghost_level=level, max_steps=500
            )

        self.reset_episode()
        print(f"Switched to: {self.game_names[game_idx]} (Level {level})")

    def reset_episode(self):
        """Reset current game"""
        self.state = self.game.reset()
        self.observer.reset()
        self.done = False

    def draw_game(self):
        """Draw current game state to surface"""
        self.screen.fill((0, 0, 0))

        game_idx = game_state['current_game']

        # Draw game-specific graphics
        if game_idx == 0:
            self.draw_snake()
        elif game_idx == 1:
            self.draw_dungeon()
        elif game_idx == 2:
            self.draw_local_view()
        elif game_idx == 3:
            self.draw_pacman()

        # Draw info panel
        self.draw_info()

    def draw_snake(self):
        """Draw Snake game"""
        cell_size = 25

        # Boundaries
        for i in range(20):
            pygame.draw.rect(self.screen, (0, 100, 0),
                           (i * cell_size, 0, cell_size, cell_size))
            pygame.draw.rect(self.screen, (0, 100, 0),
                           (i * cell_size, 19 * cell_size, cell_size, cell_size))
            pygame.draw.rect(self.screen, (0, 100, 0),
                           (0, i * cell_size, cell_size, cell_size))
            pygame.draw.rect(self.screen, (0, 100, 0),
                           (19 * cell_size, i * cell_size, cell_size, cell_size))

        # Obstacles
        for ox, oy in self.game.central_obstacles:
            pygame.draw.rect(self.screen, (255, 255, 0),
                           (ox * cell_size, oy * cell_size, cell_size, cell_size))

        # Snake body
        for x, y in self.game.snake[1:]:
            pygame.draw.rect(self.screen, (0, 255, 0),
                           (x * cell_size + 2, y * cell_size + 2,
                            cell_size - 4, cell_size - 4))

        # Snake head
        hx, hy = self.game.snake[0]
        pygame.draw.rect(self.screen, (0, 255, 255),
                       (hx * cell_size + 1, hy * cell_size + 1,
                        cell_size - 2, cell_size - 2))

        # Food
        for fx, fy in self.game.food_positions:
            pygame.draw.circle(self.screen, (255, 0, 0),
                             (int((fx + 0.5) * cell_size),
                              int((fy + 0.5) * cell_size)),
                             int(cell_size * 0.4))

    def draw_dungeon(self):
        """Draw Dungeon game"""
        cell_size = 25

        # Walls
        for wx, wy in self.game.walls:
            pygame.draw.rect(self.screen, (64, 64, 64),
                           (wx * cell_size, wy * cell_size, cell_size, cell_size))

        # Treasures
        for tx, ty in self.game.treasures:
            pygame.draw.circle(self.screen, (255, 215, 0),
                             (int((tx + 0.5) * cell_size),
                              int((ty + 0.5) * cell_size)), 8)

        # Enemies
        for enemy in self.game.enemies:
            ex, ey = enemy['pos']
            pygame.draw.rect(self.screen, (255, 0, 0),
                           (ex * cell_size + 2, ey * cell_size + 2,
                            cell_size - 4, cell_size - 4))

        # Player
        px, py = self.game.player_pos
        pygame.draw.circle(self.screen, (0, 255, 0),
                         (int((px + 0.5) * cell_size),
                          int((py + 0.5) * cell_size)),
                         int(cell_size * 0.4))

    def draw_local_view(self):
        """Draw Local View game (simplified)"""
        cell_size = 20

        # Just draw a simple version for now
        ax, ay = self.game.agent_pos
        pygame.draw.circle(self.screen, (0, 255, 0),
                         (250, 250), 10)

        # Draw some coins nearby
        for cx, cy in list(self.game.coins)[:10]:
            pygame.draw.circle(self.screen, (255, 255, 0),
                             (250 + (cx - ax) * 10, 250 + (cy - ay) * 10), 5)

    def draw_pacman(self):
        """Draw PacMan game"""
        cell_size = 25

        # Walls
        for wx, wy in self.game.walls:
            pygame.draw.rect(self.screen, (0, 0, 255),
                           (wx * cell_size, wy * cell_size, cell_size, cell_size))

        # Pellets
        for px, py in self.game.pellets:
            pygame.draw.circle(self.screen, (255, 255, 255),
                             (int((px + 0.5) * cell_size),
                              int((py + 0.5) * cell_size)), 3)

        # Ghosts
        for ghost in self.game.ghosts:
            gx, gy = ghost['pos']
            pygame.draw.circle(self.screen, (255, 0, 0),
                             (int((gx + 0.5) * cell_size),
                              int((gy + 0.5) * cell_size)),
                             int(cell_size * 0.4))

        # PacMan
        pmx, pmy = self.game.pacman_pos
        pygame.draw.circle(self.screen, (255, 255, 0),
                         (int((pmx + 0.5) * cell_size),
                          int((pmy + 0.5) * cell_size)),
                         int(cell_size * 0.45))

    def draw_info(self):
        """Draw info panel"""
        x = 550
        y = 20

        # Title
        title = self.title_font.render("Faith Multi-Game Agent", True, (0, 255, 255))
        self.screen.blit(title, (x, y))
        y += 50

        # Current game
        game_name = self.game_names[game_state['current_game']]
        text = self.font.render(f"Game: {game_name}", True, (255, 255, 255))
        self.screen.blit(text, (x, y))
        y += 35

        text = self.font.render(f"Difficulty: {game_state['difficulty']}", True, (255, 255, 255))
        self.screen.blit(text, (x, y))
        y += 35

        text = self.font.render(f"Paused: {game_state['paused']}", True, (255, 255, 0) if game_state['paused'] else (128, 128, 128))
        self.screen.blit(text, (x, y))
        y += 50

        # Stats
        text = self.font.render(f"Episodes: {game_state['episodes']}", True, (255, 255, 255))
        self.screen.blit(text, (x, y))
        y += 30

        text = self.font.render(f"Victories: {game_state['victories']}", True, (0, 255, 0))
        self.screen.blit(text, (x, y))
        y += 30

        text = self.font.render(f"Deaths: {game_state['deaths']}", True, (255, 0, 0))
        self.screen.blit(text, (x, y))
        y += 30

        if game_state['episodes'] > 0:
            win_rate = (game_state['victories'] / game_state['episodes']) * 100
            text = self.font.render(f"Win Rate: {win_rate:.1f}%", True, (0, 255, 255))
            self.screen.blit(text, (x, y))

    def step(self):
        """Run one game step"""
        if game_state['paused'] or self.done:
            return

        # Always observe state for temporal tracking
        obs = self.observer.observe(self.state)

        # Get action based on mode
        if game_state['human_playing']:
            # Human mode - use manual input
            action = game_state.get('manual_action', None)
            if action is None:
                return  # Wait for human input
            # Clear the action after using it
            game_state['manual_action'] = None
        else:
            # AI mode - get action from agent
            # Use appropriate context for each game
            # Snake uses "snake" context [1.0, 0.0, 0.0]
            # Other games use "balanced" context [0.0, 1.0, 0.0]
            if game_state['current_game'] == 0:  # Snake
                context = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            else:  # Dungeon, Local View, PacMan
                context = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

            obs_with_context = add_context_to_observation(obs, context)
            action = self.agent.get_action(obs_with_context, epsilon=0.0)

        # Step game
        self.state, reward, self.done = self.game.step(action)

        if self.done:
            game_state['episodes'] += 1

            # Check victory
            game_idx = game_state['current_game']
            victory = False
            if game_idx == 0:
                victory = self.game.total_collected >= 40
            elif game_idx == 1:
                victory = len(self.game.treasures) == 0
            elif game_idx == 2:
                victory = len(self.game.coins) == 0
            elif game_idx == 3:
                victory = len(self.game.pellets) == 0

            if victory:
                game_state['victories'] += 1
            else:
                game_state['deaths'] += 1

            # Auto-reset
            pygame.time.wait(2000)
            self.reset_episode()

    def get_frame(self):
        """Get current frame as PIL Image"""
        # Draw game
        self.draw_game()

        # Convert pygame surface to PIL Image
        img_str = pygame.image.tobytes(self.screen, 'RGB')
        img = Image.frombytes('RGB', (self.width, self.height), img_str)
        return img


# Global renderer
renderer = None


def game_loop():
    """Main game loop running in background thread"""
    global renderer

    model_path = "checkpoints/multi_game_enhanced_20251121_190832_policy.pth"
    renderer = HeadlessGameRenderer(model_path)

    print("Game loop started")

    while game_state['running']:
        try:
            renderer.step()

            # Store frame
            with game_state['lock']:
                game_state['frame'] = renderer.get_frame()

            pygame.time.wait(int(1000 / 15))  # 15 FPS
        except Exception as e:
            print(f"Game loop error: {e}")
            import traceback
            traceback.print_exc()


def generate_frames():
    """Generate MJPEG stream"""
    while True:
        with game_state['lock']:
            if game_state['frame'] is not None:
                # Convert PIL image to JPEG
                buffer = io.BytesIO()
                game_state['frame'].save(buffer, format='JPEG', quality=85)
                frame_bytes = buffer.getvalue()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        pygame.time.wait(33)  # ~30 FPS stream


@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/control')
def control():
    """Handle control commands"""
    action = request.args.get('action')

    if action == 'switch_game':
        game_id = int(request.args.get('game', 0))
        game_state['current_game'] = game_id
        if renderer:
            renderer.create_game()

    elif action == 'difficulty':
        delta = int(request.args.get('delta', 0))
        game_state['difficulty'] = max(0, game_state['difficulty'] + delta)
        if renderer:
            renderer.create_game()

    elif action == 'pause':
        game_state['paused'] = not game_state['paused']

    elif action == 'reset':
        if renderer:
            renderer.reset_episode()

    elif action == 'toggle_mode':
        game_state['human_playing'] = not game_state['human_playing']
        return jsonify({
            'status': 'ok',
            'human_playing': game_state['human_playing']
        })

    elif action == 'manual':
        direction = int(request.args.get('direction', 0))
        game_state['manual_action'] = direction

    return jsonify({'status': 'ok'})


@app.route('/stats')
def stats():
    """Get current stats"""
    return jsonify({
        'current_game': game_state['current_game'],
        'difficulty': game_state['difficulty'],
        'episodes': game_state['episodes'],
        'victories': game_state['victories'],
        'deaths': game_state['deaths'],
        'paused': game_state['paused'],
        'human_playing': game_state['human_playing']
    })


if __name__ == '__main__':
    # Start game loop in background thread
    game_thread = Thread(target=game_loop, daemon=True)
    game_thread.start()

    print("=" * 80)
    print("Web Game Streamer Starting")
    print("=" * 80)
    print("Open your browser to: http://localhost:7860")
    print("=" * 80)

    # Run Flask server
    app.run(host='0.0.0.0', port=7860, debug=False, threaded=True)

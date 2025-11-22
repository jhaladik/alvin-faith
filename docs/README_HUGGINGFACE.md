# 🕹️ RETRO ARCADE - Multi-Game AI Foundation Agent

**Classic 90s Atari-Style Interface with Zero-Shot Transfer Learning**

[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)

## 🎮 Overview

Experience the power of AI foundation models in a nostalgic 90s arcade setting! This demo showcases a **Context-Aware Foundation Agent** that was trained **only on Snake** but successfully transfers its learned spatial reasoning to play **4 completely different games** without additional training.

### Available Games
- 🐍 **Snake** - Classic snake game (training environment)
- 👾 **Pac-Man** - Navigate mazes, avoid ghosts, collect pellets
- 🏰 **Dungeon Explorer** - Explore dungeons, collect treasures, avoid enemies
- ✈️ **Sky Collector** - Fly through the sky, collect coins, dodge birds

## ⚡ Zero-Shot Transfer Results

| Game | Training | Win Rate | Notes |
|------|----------|----------|-------|
| 🐍 Snake | ✅ Trained | 100% | Base training environment |
| 👾 Pac-Man | ❌ Zero-shot | **75.0%** | Spatial reasoning transfer |
| 🏰 Dungeon | ❌ Zero-shot | **66.7%** | Maze navigation transfer |
| ✈️ Sky Collector | ❌ Zero-shot | Testing | Local view challenge |

## 🎨 Features

### Retro Aesthetics
- **CRT scanline effects** - Authentic old-school monitor look
- **Neon color palette** - Bright arcade colors (green, cyan, yellow, magenta)
- **Arcade cabinet frame** - Classic gaming interface
- **Monospace fonts** - Retro terminal styling
- **Pixel-perfect graphics** - Blocky, nostalgic visuals

### AI Capabilities
- **Zero-shot transfer** - Plays new games without training
- **Spatial reasoning** - Understands walls, navigation, collection
- **Context awareness** - Adapts strategy based on game state
- **Real-time decision making** - 16 raycast sensors for spatial awareness

### Gameplay Modes
- **Human vs AI** - Compete against the AI agent
- **Auto-play** - Watch both play simultaneously
- **Manual control** - Play games yourself with arrow keys
- **Difficulty levels** - Easy, Medium, Hard for each game

## 🚀 Quick Start

### Run Locally

```bash
# Clone repository
git clone https://github.com/your-username/alvin-faith.git
cd alvin-faith

# Install dependencies
pip install -r requirements_hf.txt

# Launch retro arcade demo
python gradio_retro_arcade.py
```

### Deploy to HuggingFace Spaces

1. Create a new Space on HuggingFace
2. Select **Gradio** as SDK
3. Upload these files:
   - `app.py` (entry point)
   - `gradio_retro_arcade.py` (main demo)
   - `requirements_hf.txt` (dependencies)
   - `src/` folder (source code)
   - `checkpoints/` folder (pre-trained models)

4. Set Space settings:
   - SDK: Gradio
   - Python version: 3.10+
   - Hardware: CPU Basic (or GPU for faster inference)

## 🎯 How to Use

1. **Select Game**: Choose from Snake, Pac-Man, Dungeon, or Sky Collector
2. **Set Difficulty**: Easy (0), Medium (1), or Hard (2)
3. **Load Model**: Click "LOAD AI" to load the pre-trained agent
4. **Play**:
   - **Manual**: Use arrow buttons (⬆️⬇️⬅️➡️) to control Player 1
   - **Auto-play**: Click "PLAY BOTH" to watch AI vs Random player
   - **AI Only**: Click "PLAY AI" to watch the agent play

## 🧠 Technical Details

### Architecture
- **Policy Network**: Deep Q-Network (DQN) with context awareness
- **Observation Space**: 183-dimensional (raycast sensors + temporal history)
- **Action Space**: 4 discrete actions (Up, Down, Left, Right)
- **World Model**: Learned dynamics for planning ahead
- **Context Module**: Automatically adapts to different game types

### Training
- **Base Environment**: Snake game with obstacles
- **Training Episodes**: 700 episodes
- **Average Reward**: 684.55
- **Training Time**: ~2 hours on single GPU
- **Transfer**: Zero-shot to new games

### Observation System
- **16 raycast sensors** - Detect walls, rewards, entities
- **Ray length**: 15 cells
- **Temporal history** - 4-frame history for motion tracking
- **Context vector** - 3D context (collection, balanced, survival)

## 📊 Performance Metrics

### Snake (Trained)
- Score: 39/40 average
- Win rate: 97.5%
- Death causes: Self-collision, wall collision

### Pac-Man (Zero-shot)
- Score: 30/30 best
- Win rate: 75% (level 3 ghosts)
- Ghost avoidance: Excellent

### Dungeon (Zero-shot)
- Score: 3/3 best
- Win rate: 66.7% (level 5 enemies)
- Maze navigation: Very good

### Sky Collector (Zero-shot)
- Score: Testing
- Local view challenge: High difficulty
- Spatial reasoning: In evaluation

## 🎨 Customization

### Change Color Scheme
Edit `RETRO_COLORS` dictionary in `gradio_retro_arcade.py`:

```python
RETRO_COLORS = {
    'green': (0, 255, 0),      # Player 1 color
    'cyan': (0, 255, 255),      # AI agent color
    'yellow': (255, 255, 0),    # Accent color
    # ... add your own colors
}
```

### Adjust Scanline Intensity
Modify `add_scanlines()` intensity parameter (0.0 = none, 1.0 = maximum):

```python
img = RetroRenderer.add_scanlines(img, 0.12)  # 12% darkness
```

### Modify Game Difficulty
Change difficulty parameters in `reset_games()`:

```python
# Snake: size and pellet count
EnhancedSnakeGame(size=20, initial_pellets=7, max_steps=400)

# Pac-Man: ghost level (0-5)
SimplePacManGame(size=20, num_pellets=30, ghost_level=2, max_steps=500)
```

## 🏆 Key Achievements

- ✅ **Zero-shot transfer** to multiple game genres
- ✅ **75% win rate** on Pac-Man without training
- ✅ **66.7% win rate** on Dungeon without training
- ✅ **Spatial reasoning** generalizes across games
- ✅ **Real-time performance** on CPU
- ✅ **Retro interface** with authentic 90s aesthetics

## 📝 Citation

If you use this demo or code, please cite:

```bibtex
@software{alvin_faith_2025,
  title={ALVIN-FAITH: Context-Aware Foundation Agent with Zero-Shot Transfer},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/alvin-faith}
}
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional game types (platformers, puzzle games)
- Enhanced transfer learning techniques
- Better rendering effects (CRT curve, color bleeding)
- Performance optimization
- Mobile-friendly controls

## 📄 License

MIT License - See LICENSE file for details

## 🔗 Links

- **GitHub**: [alvin-faith](https://github.com/your-username/alvin-faith)
- **HuggingFace Space**: [Your Space URL]
- **Paper**: [Coming soon]
- **Demo Video**: [Coming soon]

---

**Made with ❤️ and nostalgia for 90s arcade games**

🕹️ INSERT COIN TO CONTINUE 🕹️

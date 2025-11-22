---
title: Faith Multi-Game Agent
emoji: 🎮
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Faith Multi-Game Agent 🎮

**One AI agent, four different games!** Watch a single reinforcement learning agent play Snake, Dungeon, Local View, and PacMan through your browser using VNC.

## 🎯 Features

- **Multi-Game Switching**: Switch between 4 games in real-time
- **Zero-Shot Transfer**: Agent trained on Snake, tested on other games
- **Browser-Based**: Play through noVNC web interface
- **Adjustable Difficulty**: Change game difficulty on the fly

## 🕹️ Games

1. **Snake** - Navigate obstacles and collect food (trained)
2. **Dungeon** - Explore maze, collect treasures, avoid enemies (zero-shot)
3. **Local View** - Moving camera perspective in large world (zero-shot)
4. **PacMan** - Collect pellets, avoid ghosts (zero-shot)

## 🚀 How to Use

1. Wait for the Space to load (may take 1-2 minutes on first start)
2. The VNC interface will appear automatically in the iframe above
3. Click **"Connect"** in the noVNC window
4. Watch the agent play! Use keyboard controls to switch games

## 🎮 Controls

Once connected to the VNC interface:

- **1** - Switch to Snake
- **2** - Switch to Dungeon
- **3** - Switch to Local View
- **4** - Switch to PacMan
- **UP/DOWN** - Increase/Decrease difficulty level
- **SPACE** - Pause/Resume
- **R** - Reset current episode
- **ESC** - Quit

## 🧠 Technical Details

- **Architecture**: Context-Aware DQN with temporal observer
- **Observation**: 16-ray raycasting + temporal history (183 dims)
- **Actions**: 4 directions (Up, Down, Left, Right)
- **Training**: Multi-game curriculum learning on Snake
- **Transfer**: Spatial reasoning transfers to unseen games

## 📊 Model Info

- **Checkpoint**: `multi_game_enhanced_20251121_190832_policy.pth`
- **Trained on**: Snake with obstacles (Level 2)
- **Zero-shot games**: Dungeon, Local View, PacMan
- **Framework**: PyTorch (CPU inference)

## 🛠️ Tech Stack

- **Backend**: Docker container with Xvfb (virtual display)
- **VNC**: x11vnc + noVNC (browser VNC client)
- **Game Engine**: Pygame
- **ML**: PyTorch (CPU-only)

## ⚠️ Notes

- First load takes ~1-2 minutes to initialize services
- ALSA audio warnings are harmless (no sound card in container)
- If connection fails, try refreshing the page
- Game runs at 15 FPS for smooth browser experience

## 📝 Citation

```bibtex
@software{faith_multi_game_2025,
  title={Faith: Context-Aware Foundation Agent for Multi-Game Transfer},
  author={Jan Haladik},
  year={2025},
  url={https://github.com/jhaladik/alvin-faith}
}
```

## 🔗 Links

- **GitHub**: [alvin-faith](https://github.com/jhaladik/alvin-faith)
- **Paper**: Coming soon
- **Demo Video**: Coming soon

---

**Enjoy watching the agent's spatial reasoning transfer across games!** 🚀

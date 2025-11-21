# 🤖 Alvin Faith - Context-Aware Foundation Agent with Faith-Based Exploration

A reinforcement learning agent that combines **context-aware behavior adaptation** with **faith-based exploration** to discover hidden mechanics and transfer knowledge across environments.

## 🌟 Key Features

### 1. **Faith-Based Exploration** 🔮
Persistent exploration despite negative feedback - discovers hidden mechanics through deliberate experimentation
- **Faith Patterns**: Evolutionary population of exploration strategies
- **Entity Discovery**: Learns entity types without labels
- **Pattern Transfer**: Game-agnostic strategy transfer
- **Mechanic Detection**: Discovers hidden rules automatically

### 2. **Context-Aware Adaptation** 🎯
Automatically adapts behavior based on environmental context:
- **Snake Mode** (0 entities): Aggressive collection strategy
- **Balanced Mode** (2-3 entities): Tactical gameplay
- **Survival Mode** (4+ entities): Cautious avoidance

### 3. **Model-Based Planning** 🧠
Plans 20 steps ahead using learned world model for proactive decision-making

## 📊 Performance Highlights

| Metric | Value |
|--------|-------|
| **Training Episodes** | 700 |
| **Average Reward** | 684.55 |
| **Pac-Man Score** | 17.62 avg (50 episodes) |
| **Warehouse Discovery** | 80% (4/5 hidden mechanics) |
| **Zero-Shot Transfer** | ✅ Success |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/jhaladik/alvin-faith.git
cd alvin-faith

# Install dependencies
pip install -r requirements.txt
```

### Test with Pre-trained Model

```bash
# Run comprehensive test suite
python src/test_expanded_faith.py checkpoints/faith_fixed_20251120_162417_final_policy.pth --episodes 50

# Visual demo with pygame
python src/faith_visual_games.py checkpoints/faith_fixed_20251120_162417_final_policy.pth --speed 10

# Launch Gradio web demo
python gradio_demo.py
```

### Train New Model

```bash
# Train with fixed world model (recommended)
python src/train_expanded_faith_fixed.py --episodes 1000 --log-every 100

# Train on Snake only (focused training)
python src/train_snake_focused.py --episodes 500
```

## 🏗️ Architecture

### Agent Model
- **Input**: 183 dimensions (180 temporal + 3 context)
- **Observer**: 16 rays × 15 tiles = 240 spatial features
- **Temporal**: Micro (5) + Meso (20) + Macro (50) frames
- **Network**: Hierarchical DQN with 4 specialized heads
- **Parameters**: ~150K trainable

### World Model (Fixed)
- **Predicts**: 180-dim observations (NOT 183 - bottleneck fixed!)
- **Context**: Passed through unchanged
- **Architecture**: Context-aware predictor
- **Purpose**: Enables 20-step lookahead planning

### Faith System
- **Population**: 20 evolutionary patterns
- **Behaviors**: Wait, explore, rhythmic, sacrificial
- **Fitness**: Evolves based on discovery success
- **Discovery Rate**: Tracks novel mechanic findings

## 🎮 Supported Environments

### Arcade Games
- **Snake**: Pure collection (10 pellets)
- **Pac-Man**: Balanced threats + rewards
- **Dungeon**: High-threat survival + treasure

### Warehouse Scenarios
- **Hidden Shortcut**: Supervisor-dependent paths
- **Charging Station**: Time-dependent recharging
- **Priority Zones**: Color-coded package priorities

## 📁 Repository Structure

```
alvin-faith/
├── src/
│   ├── train_expanded_faith_fixed.py      # PRIMARY: Fixed world model training
│   ├── train_snake_focused.py             # Snake-only focused training
│   ├── test_expanded_faith.py             # Comprehensive test suite
│   ├── faith_visual_games.py              # Visual pygame demo
│   ├── context_aware_agent.py             # Agent architecture
│   └── core/
│       ├── expanded_temporal_observer.py  # 180-dim observer
│       ├── context_aware_world_model.py   # Fixed world model
│       ├── faith_system.py                # Faith patterns & evolution
│       ├── entity_discovery.py            # Entity classification
│       ├── pattern_transfer.py            # Universal patterns
│       ├── mechanic_detectors.py          # Hidden mechanic discovery
│       ├── planning_test_games.py         # Snake, Pac-Man, Dungeon
│       └── ... (other core modules)
├── checkpoints/
│   ├── faith_fixed_20251120_162417_final_*.pth      # Production model
│   └── faith_evolution_20251120_194555_best_*.pth   # Best performance
├── gradio_demo.py                         # HuggingFace Spaces demo
├── requirements.txt                       # Dependencies
└── README.md                              # This file
```

## 🔬 Key Innovations

### 1. Fixed World Model Bottleneck
**Problem**: Old model predicted 183 dims (180 obs + 3 context) - wasting capacity on constant context values

**Solution**: Context-aware architecture predicts only 180 dims (observations), passes context through unchanged

**Result**: Faster convergence, better planning, clearer gradients

### 2. Expanded Temporal Observer
- **2x spatial coverage**: 16 rays × 15 tiles (vs 8 rays × 10 tiles)
- **Multi-scale temporal**: Micro + Meso + Macro frames
- **Ghost behavior detection**: Chase/scatter/random modes
- **180-dim rich observations**: Comprehensive world state

### 3. Faith-Based Exploration
Traditional RL gives up after negative feedback. Faith persists:
- **Persistent trials**: Continues exploring despite penalties
- **Hypothesis formation**: Builds models of hidden mechanics
- **Evolutionary fitness**: Best patterns survive and reproduce
- **Discovery tracking**: Records novel findings

## 📈 Training Details

### Mixed Scenario Training
```
Context Distribution:
  SNAKE (0 entities):      30%  - Pure collection
  BALANCED (2-3 entities): 50%  - Tactical gameplay
  SURVIVAL (4+ entities):  20%  - High-threat avoidance
```

### Hyperparameters
- **Episodes**: 700 (production model)
- **Batch Size**: 64
- **Learning Rate**: 1e-4
- **Gamma**: 0.99
- **Epsilon**: 1.0 → 0.01 (decay over 350 episodes)
- **Planning Frequency**: 20%
- **Planning Horizon**: 20 steps
- **Faith Frequency**: 30%

## 🎯 Action Distribution

During gameplay, the agent chooses actions from three sources:

1. **Faith** (0-30%): Exploration via evolutionary patterns
2. **Planning** (0-20%): Model-based lookahead
3. **Reactive** (50-100%): Direct policy decisions

Visual indicators:
- 🟣 **Magenta agent** = Faith action
- 🔵 **Cyan agent** = Planning action
- 🟡 **Yellow agent** = Reactive action

## 🧪 Testing

### Run Test Suite
```bash
# All games (Snake, Pac-Man, Dungeon)
python src/test_expanded_faith.py checkpoints/faith_fixed_20251120_162417_final_policy.pth

# Specific game
python src/test_expanded_faith.py checkpoints/faith_fixed_20251120_162417_final_policy.pth --game pacman --episodes 100

# With faith analysis
python src/test_expanded_faith.py checkpoints/faith_fixed_20251120_162417_final_policy.pth --analyze-faith
```

### Visual Demo
```bash
# Launch pygame demo
python src/faith_visual_games.py checkpoints/faith_fixed_20251120_162417_final_policy.pth --speed 10

# Controls:
#   1 - Switch to Snake
#   2 - Switch to Pac-Man
#   3 - Switch to Dungeon
#   SPACE - Toggle AI/Manual
#   R - Reset game
#   ESC - Quit
```

## 🌐 Web Demo

Launch interactive Gradio demo:

```bash
python gradio_demo.py
```

Then open http://localhost:7860 in your browser.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{alvin_faith_2024,
  title={Alvin Faith: Context-Aware Foundation Agent with Faith-Based Exploration},
  author={Jan Haladik},
  year={2024},
  url={https://github.com/jhaladik/alvin-faith}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test changes thoroughly
4. Submit a pull request

## 🔗 Related Work

- **Context-Aware DQN**: Solves spurious correlation in transfer learning
- **Temporal Observer**: Multi-scale observation system
- **World Models**: Learned dynamics for planning
- **Evolutionary Strategies**: Population-based exploration

## 📧 Contact

For questions or issues, please open a GitHub issue at:
https://github.com/jhaladik/alvin-faith/issues

---

**Ready to explore?**

```bash
python src/faith_visual_games.py checkpoints/faith_fixed_20251120_162417_final_policy.pth
```

Watch the agent discover hidden mechanics through faith-based exploration! 🔮✨

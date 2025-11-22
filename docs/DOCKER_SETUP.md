# Docker Setup for Pygame Generalization Test

This guide shows how to run `test_generalization.py` in a Docker container with VNC access through your browser.

## Prerequisites

- Docker installed on your machine
- Docker Compose (optional, for easier setup)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

2. **Open your browser:**
   Navigate to: http://localhost:7860/vnc.html

3. **Click "Connect"** in the noVNC interface

4. **You should see the pygame window!** Use your keyboard to control:
   - UP/DOWN arrows: Change obstacle level
   - SPACE: Pause/Resume
   - R: Reset episode
   - ESC: Quit

5. **Stop the container:**
   Press `Ctrl+C` in the terminal

### Option 2: Using Docker Commands

1. **Build the Docker image:**
   ```bash
   docker build -t snake-game .
   ```

2. **Run the container:**
   ```bash
   docker run -p 7860:7860 snake-game
   ```

3. **Open browser and connect** (same as above)

4. **Stop the container:**
   ```bash
   docker ps  # Find container ID
   docker stop <container_id>
   ```

## Configuration

### Change Model Path

Edit `start.sh` line 32 to use a different checkpoint:
```bash
MODEL_PATH="checkpoints/your_model_name.pth"
```

### Change Game Settings

Edit `start.sh` line 43 to modify parameters:
```bash
python test_generalization.py --model "$MODEL_PATH" --obstacle-level 5 --speed 15
```

Parameters:
- `--model`: Path to model checkpoint (required)
- `--obstacle-level`: Number of obstacle level (default: 3)
- `--speed`: Game speed in FPS (default: 5)

## Troubleshooting

### "Model checkpoint not found"
Make sure you have the model checkpoint in the `checkpoints/` directory:
```bash
ls -la checkpoints/
```

### "Cannot connect to display"
The container may need more time to start Xvfb. The start script includes sleep delays, but you can increase them in `start.sh` if needed.

### Port 7860 already in use
Change the port in `docker-compose.yml`:
```yaml
ports:
  - "8080:7860"  # Use port 8080 instead
```
Then access: http://localhost:8080/vnc.html

### Black screen in browser
Wait a few seconds for the X server to fully initialize, then refresh the browser.

## How It Works

1. **Xvfb** creates a virtual display (no physical monitor needed)
2. **x11vnc** shares the virtual display via VNC protocol
3. **noVNC** provides a web-based VNC client in your browser
4. **pygame** runs normally, thinking it has a real display

## Deploy to Hugging Face Spaces

Once you've tested locally, you can deploy to HF Spaces:

1. Create a new Space on Hugging Face
2. Select "Docker" as SDK
3. Push these files:
   - `Dockerfile`
   - `start.sh`
   - `test_generalization.py`
   - `requirements.txt`
   - `checkpoints/` (your model files)
   - All `src/` files
4. Add to your Space's README.md:
   ```yaml
   ---
   title: Snake Generalization Test
   emoji: 🐍
   colorFrom: green
   colorTo: blue
   sdk: docker
   app_port: 7860
   ---
   ```

The Space will be accessible at: https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

## File Structure

```
alvin-faith/
├── Dockerfile              # Docker container definition
├── docker-compose.yml      # Docker Compose configuration
├── start.sh               # Startup script for VNC and game
├── test_generalization.py # The pygame application
├── requirements.txt       # Python dependencies
├── checkpoints/           # Model checkpoints
│   └── snake_context_aware_dqn_final.pth
└── src/                   # Source code
    └── ...
```

## Notes

- The container stays running even after the game exits (for debugging)
- Keyboard input works through the browser VNC interface
- The game runs at the speed specified in `start.sh` (default: 10 FPS)

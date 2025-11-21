"""
Gradio Demo for Context-Aware Foundation Agent
HuggingFace Spaces Deployment

Interactive demonstration of faith-based exploration, planning, and context adaptation.
Showcases Pac-Man gameplay and warehouse scenario discovery.
"""

import gradio as gr
import numpy as np
import torch
import sys
import os
import time
from PIL import Image, ImageDraw, ImageFont
import io

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "core"))

from src.context_aware_agent import ContextAwareDQN, infer_context_from_observation, add_context_to_observation
from src.core.expanded_temporal_observer import ExpandedTemporalObserver
from src.core.planning_test_games import PacManGame, SnakeGame
from src.core.context_aware_world_model import ContextAwareWorldModel
from src.warehouse_faith_scenarios import create_scenario

# Colors (High contrast for visibility)
COLORS = {
    'floor': (40, 40, 45),  # Dark floor for contrast
    'background': (30, 30, 35),  # Darker background
    'wall': (80, 80, 90),
    'border': (100, 100, 110),
    'agent': (255, 220, 0),  # Bright Yellow
    'agent_faith': (255, 0, 255),  # Bright Magenta (faith action)
    'agent_planning': (0, 255, 255),  # Bright Cyan (planning action)
    'food': (50, 255, 50),  # Bright green
    'ghost': (255, 80, 80),  # Bright red
    'package': (100, 200, 255),  # Bright blue
    'shelf': (139, 90, 60),
    'worker': (255, 150, 100),  # Brighter worker
    'shortcut_active': (0, 255, 0),
    'shortcut_blocked': (255, 0, 0),
    'priority_red': (255, 70, 70),
    'priority_green': (70, 255, 70),
    'priority_blue': (70, 180, 255),
}


class GradioGameDemo:
    """Interactive game demo for Gradio"""

    def __init__(self, model_path='src/checkpoints/faith_fixed_20251120_162417_final_policy.pth'):
        self.model_path = model_path
        self.agent = None
        self.world_model = None
        self.observer = ExpandedTemporalObserver(num_rays=16, ray_length=15, verbose=False)  # Always create observer
        self.game = None
        self.scenario = None
        self.game_state = None
        self.episode_steps = 0
        self.episode_reward = 0
        self.action_counts = {'faith': 0, 'planning': 0, 'reactive': 0}
        self.last_action_type = 'reactive'
        self.discoveries = []

        self.load_model()

    def load_model(self):
        """Load trained model"""
        if not os.path.exists(self.model_path):
            print(f"Warning: Model not found at {self.model_path}")
            print(f"  Demo will run in random action mode")
            return

        try:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

            # Load policy
            self.agent = ContextAwareDQN(obs_dim=183, action_dim=4)
            self.agent.load_state_dict(checkpoint['policy_net'])
            self.agent.eval()

            # Load world model
            base_path = self.model_path.replace('_policy.pth', '').replace('_final', '').replace('_best', '')
            world_model_path = f"{base_path}_world_model.pth"

            if os.path.exists(world_model_path):
                wm_checkpoint = torch.load(world_model_path, map_location='cpu', weights_only=False)
                obs_dim = checkpoint.get('world_model_obs_dim', 180)
                context_dim = checkpoint.get('world_model_context_dim', 3)

                self.world_model = ContextAwareWorldModel(
                    obs_dim=obs_dim,
                    context_dim=context_dim,
                    action_dim=4,
                    hidden_dim=256
                )
                self.world_model.load_state_dict(wm_checkpoint['model'])
                self.world_model.eval()

            print(f"Model loaded: {len(checkpoint.get('episode_rewards', []))} episodes trained")
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"  Demo will run in random action mode")
            self.agent = None
            self.world_model = None

    def reset_game(self, game_type, scenario_name=None):
        """Reset game/scenario"""
        self.episode_steps = 0
        self.episode_reward = 0
        self.action_counts = {'faith': 0, 'planning': 0, 'reactive': 0}
        self.discoveries = []

        if game_type == 'pacman':
            self.game = PacManGame(size=20)
            self.scenario = None
            self.game_state = self.game.reset()
        elif game_type == 'snake':
            self.game = SnakeGame(size=20)
            self.scenario = None
            self.game_state = self.game.reset()
        else:  # warehouse
            self.scenario = create_scenario(scenario_name or 'hidden_shortcut', size=20)
            self.game = None
            self.game_state = self.scenario.reset()

        self.observer.reset()
        return self.render(), self.get_info_html()

    def get_action(self, faith_freq, planning_freq, planning_horizon):
        """Get action from agent"""
        if self.game:
            obs = self.observer.observe(self.game._get_game_state())
        else:
            obs = self.observer.observe(self.game_state)

        context = infer_context_from_observation(obs)
        obs_with_context = add_context_to_observation(obs, context)

        # If no agent loaded, use random actions
        if self.agent is None:
            action = np.random.randint(4)
            self.last_action_type = 'random'
            self.action_counts['reactive'] += 1  # Count as reactive for display
            return action

        # Decide action type
        rand = np.random.random()

        if rand < faith_freq:
            # Faith action (random exploration)
            action = np.random.randint(4)
            self.last_action_type = 'faith'
            self.action_counts['faith'] += 1
        elif self.world_model and rand < (faith_freq + planning_freq):
            # Planning action
            action = self.plan_action(obs_with_context, planning_horizon)
            self.last_action_type = 'planning'
            self.action_counts['planning'] += 1
        else:
            # Reactive action
            action = self.agent.get_action(obs_with_context, epsilon=0.0)
            self.last_action_type = 'reactive'
            self.action_counts['reactive'] += 1

        return action

    def plan_action(self, state, horizon):
        """Planning with world model"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        best_action = 0
        best_return = -float('inf')

        with torch.no_grad():
            for action in range(4):
                total_return = 0.0
                for _ in range(5):  # 5 rollouts
                    rollout_return = self.simulate_rollout(state_tensor, action, horizon)
                    total_return += rollout_return
                avg_return = total_return / 5
                if avg_return > best_return:
                    best_return = avg_return
                    best_action = action

        return best_action

    def simulate_rollout(self, state, first_action, horizon):
        """Simulate trajectory"""
        current_state = state.clone()
        total_return = 0.0
        discount = 1.0
        gamma = 0.99

        with torch.no_grad():
            action_tensor = torch.LongTensor([first_action])
            next_state, reward, done = self.world_model(current_state, action_tensor)
            total_return += reward.item() * discount
            discount *= gamma

            if done.item() > 0.5:
                return total_return

            current_state = next_state

            for _ in range(horizon - 1):
                q_values = self.agent.get_combined_q(current_state)
                action = q_values.argmax(dim=1).item()
                action_tensor = torch.LongTensor([action])
                next_state, reward, done = self.world_model(current_state, action_tensor)
                total_return += reward.item() * discount
                discount *= gamma
                if done.item() > 0.5:
                    break
                current_state = next_state

        return total_return

    def step(self, faith_freq, planning_freq, planning_horizon):
        """Take one game step"""
        action = self.get_action(faith_freq, planning_freq, planning_horizon)

        # Execute action
        if self.game:
            self.game_state, reward, done = self.game.step(action)
        else:
            self.game_state, reward, done = self.scenario.step(action)

            # Check for discoveries in warehouse
            if self.scenario and hasattr(self.scenario, 'hidden_mechanics_discovered'):
                new_discoveries = len(self.scenario.hidden_mechanics_discovered) - len(self.discoveries)
                if new_discoveries > 0:
                    self.discoveries = list(self.scenario.hidden_mechanics_discovered)

        self.episode_steps += 1
        self.episode_reward += reward

        status = ""
        if done:
            score = self.game_state.get('score', 0) if self.game else self.scenario.packages_picked
            status = f"🎉 Episode Complete! Score: {score}, Steps: {self.episode_steps}, Reward: {self.episode_reward:.1f}"

        return self.render(), self.get_info_html(), status

    def render(self):
        """Render game as PIL Image"""
        cell_size = 30
        border_size = 4

        if self.game:
            size = self.game.size
        else:
            size = self.scenario.size

        # Create image with border
        total_size = size * cell_size + 2 * border_size
        img = Image.new('RGB', (total_size, total_size), COLORS['background'])
        draw = ImageDraw.Draw(img)

        # Draw border
        draw.rectangle([0, 0, total_size-1, total_size-1], outline=COLORS['border'], width=border_size)

        # Create game area
        game_img = Image.new('RGB', (size * cell_size, size * cell_size), COLORS['floor'])
        game_draw = ImageDraw.Draw(game_img)

        if self.game:
            self._render_game(game_draw, cell_size)
        else:
            self._render_warehouse(game_draw, cell_size)

        # Paste game area into bordered image
        img.paste(game_img, (border_size, border_size))

        return img

    def _render_game(self, draw, cell_size):
        """Render Pac-Man or Snake game"""
        size = self.game.size

        # Draw walls (for both games)
        if hasattr(self.game, 'walls'):
            for (x, y) in self.game.walls:
                draw.rectangle([x * cell_size, y * cell_size,
                              (x + 1) * cell_size, (y + 1) * cell_size],
                             fill=COLORS['wall'])

        # Draw game-specific elements
        if hasattr(self.game, 'pellets'):  # Pac-Man
            # Draw pellets (food)
            for (x, y) in self.game.pellets:
                cx = int((x + 0.5) * cell_size)
                cy = int((y + 0.5) * cell_size)
                r = int(cell_size * 0.15)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLORS['food'])

            # Draw ghosts
            for ghost in self.game.ghosts:
                x, y = ghost['pos']
                cx = int((x + 0.5) * cell_size)
                cy = int((y + 0.5) * cell_size)
                r = int(cell_size * 0.35)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLORS['ghost'])

        elif hasattr(self.game, 'snake'):  # Snake
            # Draw snake body
            for (x, y) in self.game.snake:
                draw.rectangle([x * cell_size + 2, y * cell_size + 2,
                              (x + 1) * cell_size - 2, (y + 1) * cell_size - 2],
                             fill=COLORS['food'])

            # Draw food pellets
            if hasattr(self.game, 'food_positions'):
                for (fx, fy) in self.game.food_positions:
                    cx = int((fx + 0.5) * cell_size)
                    cy = int((fy + 0.5) * cell_size)
                    r = int(cell_size * 0.25)
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLORS['ghost'])

        # Draw agent (color-coded by action type)
        agent_pos = self.game_state.get('agent_pos', (size//2, size//2))
        x, y = agent_pos
        cx = int((x + 0.5) * cell_size)
        cy = int((y + 0.5) * cell_size)
        r = int(cell_size * 0.4)

        if self.last_action_type == 'faith':
            color = COLORS['agent_faith']
        elif self.last_action_type == 'planning':
            color = COLORS['agent_planning']
        else:
            color = COLORS['agent']

        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    def _render_warehouse(self, draw, cell_size):
        """Render warehouse scenario"""
        # Draw walls/shelves
        for x, y in self.scenario.walls:
            color = COLORS['shelf']

            # Color shortcuts if applicable
            if hasattr(self.scenario, 'shortcut_walls') and (x, y) in self.scenario.shortcut_walls:
                supervisor_dist = abs(x - self.scenario.supervisor['pos'][0]) + \
                                abs(y - self.scenario.supervisor['pos'][1])
                color = COLORS['shortcut_active'] if supervisor_dist > 5 else COLORS['shortcut_blocked']

            draw.rectangle([x * cell_size, y * cell_size,
                          (x + 1) * cell_size, (y + 1) * cell_size],
                         fill=color)

        # Draw packages
        for package in self.scenario.packages:
            x, y = package['pos']
            cx = int((x + 0.5) * cell_size)
            cy = int((y + 0.5) * cell_size)
            r = int(cell_size * 0.25)

            pkg_type = package.get('type', 'standard')
            if pkg_type == 'red':
                color = COLORS['priority_red']
            elif pkg_type == 'green':
                color = COLORS['priority_green']
            elif pkg_type == 'blue':
                color = COLORS['priority_blue']
            else:
                color = COLORS['package']

            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

        # Draw workers
        for worker in self.scenario.workers:
            x, y = worker['pos']
            cx = int((x + 0.5) * cell_size)
            cy = int((y + 0.5) * cell_size)
            r = int(cell_size * 0.2)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLORS['worker'])

        # Draw AGV (color-coded)
        x, y = self.scenario.agv_pos
        cx = int((x + 0.5) * cell_size)
        cy = int((y + 0.5) * cell_size)
        r = int(cell_size * 0.35)

        if self.last_action_type == 'faith':
            color = COLORS['agent_faith']
        elif self.last_action_type == 'planning':
            color = COLORS['agent_planning']
        else:
            color = COLORS['agent']

        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    def get_info_html(self):
        """Get info panel HTML"""
        if self.game:
            score = self.game_state.get('score', 0)
            title = "🎮 Pac-Man" if hasattr(self.game, 'pellets') else "🐍 Snake"
        else:
            score = self.scenario.packages_picked
            title = f"📦 {self.scenario.name}"

        total_actions = sum(self.action_counts.values()) or 1

        html = f"""
        <div style="font-family: Arial; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin-bottom: 20px; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px;">{title}</h2>

            <div style="margin: 15px 0; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                <h3 style="color: #ffeb3b; margin-bottom: 10px;">📊 Episode Stats</h3>
                <p style="font-size: 16px; margin: 5px 0;"><strong>Score:</strong> <span style="color: #4caf50; font-size: 20px;">{score}</span></p>
                <p style="font-size: 16px; margin: 5px 0;"><strong>Steps:</strong> {self.episode_steps}</p>
                <p style="font-size: 16px; margin: 5px 0;"><strong>Reward:</strong> {self.episode_reward:.1f}</p>
            </div>

            <div style="margin: 15px 0; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                <h3 style="color: #ffeb3b; margin-bottom: 10px;">🎯 Action Distribution</h3>
                <p style="margin: 5px 0;"><span style="color: #FF00FF; font-size: 20px;">●</span> <strong>Faith:</strong> {self.action_counts['faith']/total_actions*100:.1f}%</p>
                <p style="margin: 5px 0;"><span style="color: #00FFFF; font-size: 20px;">●</span> <strong>Planning:</strong> {self.action_counts['planning']/total_actions*100:.1f}%</p>
                <p style="margin: 5px 0;"><span style="color: #FFC800; font-size: 20px;">●</span> <strong>Reactive:</strong> {self.action_counts['reactive']/total_actions*100:.1f}%</p>
            </div>

            <div style="margin: 15px 0; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
                <h3 style="color: #ffeb3b; margin-bottom: 10px;">🔍 Current Action</h3>
                <p style="font-size: 24px; font-weight: bold; color: {'#FF00FF' if self.last_action_type == 'faith' else '#00FFFF' if self.last_action_type == 'planning' else '#FFC800'}; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    {self.last_action_type.upper()}
                </p>
            </div>
        """

        if self.scenario and self.discoveries:
            html += """
            <div style="margin: 15px 0; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                <h3 style="color: #ffeb3b; margin-bottom: 10px;">✨ Discoveries</h3>
            """
            for discovery in self.discoveries[-3:]:
                name = discovery['mechanic'].replace('_', ' ').title()
                html += f"<p style='color: #4caf50; margin: 5px 0; font-weight: bold;'>✓ {name}</p>"
            html += "</div>"

        html += "</div>"
        return html


# Initialize demo
demo_instance = GradioGameDemo()


def create_demo():
    """Create Gradio interface"""

    with gr.Blocks(title="Context-Aware Foundation Agent Demo", theme=gr.themes.Default()) as demo:
        gr.Markdown("""
        # 🤖 Context-Aware Foundation Agent

        **Interactive demonstration** of faith-based exploration, planning, and context adaptation.

        ### Key Features:
        - 🎯 **Context Adaptation**: Automatically switches between collection, balanced, and survival strategies
        - 🔮 **Faith-Based Exploration**: Discovers hidden mechanics through persistent exploration
        - 🧠 **Model-Based Planning**: Plans 20 steps ahead using learned world model
        - 🌐 **Zero-Shot Transfer**: 80% mechanic discovery on unseen warehouse scenarios

        ### Performance Highlights:
        - **Pac-Man**: 17.62 avg score (50 episodes)
        - **Warehouse Discovery**: 4/5 hidden mechanics discovered
        - **Training**: 700 episodes, 684.55 avg reward

        ---
        """)

        with gr.Row():
            with gr.Column(scale=2):
                game_display = gr.Image(label="Game View", type="pil", height=650)
                status_text = gr.Textbox(label="Status", lines=1, interactive=False)

            with gr.Column(scale=1):
                info_panel = gr.HTML(label="Statistics")

        with gr.Row():
            game_type = gr.Radio(
                choices=["pacman", "snake", "warehouse"],
                value="pacman",
                label="Environment"
            )
            scenario_type = gr.Dropdown(
                choices=["hidden_shortcut", "charging_station", "priority_zone"],
                value="hidden_shortcut",
                label="Warehouse Scenario",
                visible=False
            )

        with gr.Row():
            faith_freq = gr.Slider(0.0, 0.5, value=0.0, step=0.05, label="Faith Frequency (Exploration)")
            planning_freq = gr.Slider(0.0, 0.5, value=0.2, step=0.05, label="Planning Frequency")
            planning_horizon = gr.Slider(5, 30, value=20, step=5, label="Planning Horizon")

        with gr.Row():
            reset_btn = gr.Button("🔄 Reset Game", variant="secondary")
            step_btn = gr.Button("▶️ Step", variant="primary")
            run_btn = gr.Button("🎬 Run Episode", variant="primary")

        gr.Markdown("""
        ### 🎨 Visual Guide:
        - 🟣 **Magenta Agent** = Faith action (exploration)
        - 🔵 **Cyan Agent** = Planning action (model-based)
        - 🟡 **Yellow Agent** = Reactive action (policy-based)

        ### 🎮 Recommended Configurations:
        - **Best Performance**: Faith 0%, Planning 20%, Horizon 20
        - **Exploration Mode**: Faith 30%, Planning 0%, Horizon 20
        - **Balanced**: Faith 15%, Planning 15%, Horizon 20
        """)

        # Event handlers
        def update_scenario_visibility(game):
            return gr.update(visible=(game == "warehouse"))

        game_type.change(
            update_scenario_visibility,
            inputs=[game_type],
            outputs=[scenario_type]
        )

        def reset_wrapper(game, scenario):
            return demo_instance.reset_game(game, scenario)

        reset_btn.click(
            reset_wrapper,
            inputs=[game_type, scenario_type],
            outputs=[game_display, info_panel]
        )

        def step_wrapper(faith, planning, horizon):
            return demo_instance.step(faith, planning, horizon)

        step_btn.click(
            step_wrapper,
            inputs=[faith_freq, planning_freq, planning_horizon],
            outputs=[game_display, info_panel, status_text]
        )

        def run_episode(game, scenario, faith, planning, horizon):
            """Run episode with animated steps"""
            demo_instance.reset_game(game, scenario)

            for step in range(200):  # Max 200 steps
                img, info, status = demo_instance.step(faith, planning, horizon)

                # Small delay for visible animation (0.05s = 20 FPS)
                time.sleep(0.05)

                # Yield intermediate results for animation
                yield img, info, f"Step {step + 1}/200... {status}"

                if "Complete" in status:
                    yield img, info, status
                    return

            yield img, info, "⏱️ Max steps reached"

        run_btn.click(
            run_episode,
            inputs=[game_type, scenario_type, faith_freq, planning_freq, planning_horizon],
            outputs=[game_display, info_panel, status_text]
        )

        # Initialize display
        demo.load(
            lambda: demo_instance.reset_game('pacman'),
            outputs=[game_display, info_panel]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()

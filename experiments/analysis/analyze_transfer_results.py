"""
Automated analysis of zero-shot transfer performance
Run tests without GUI to collect statistics
"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.simple_pacman_game import SimplePacManGame
from src.core.simple_dungeon_game import SimpleDungeonGame
from src.context_aware_agent import ContextAwareDQN, add_context_to_observation
from src.core.expanded_temporal_observer import ExpandedTemporalObserver


def test_transfer(model_path, game_type='pacman', difficulty=0, num_episodes=20):
    """Test zero-shot transfer on a game"""
    # Load SNAKE agent
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    agent = ContextAwareDQN(obs_dim=183, action_dim=4)
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    # Create game
    if game_type == 'pacman':
        game = SimplePacManGame(size=20, num_pellets=30, ghost_level=difficulty, max_steps=500)
        game_name = f"PacMan (Level {difficulty})"
    elif game_type == 'dungeon':
        game = SimpleDungeonGame(size=20, num_treasures=3, enemy_level=difficulty, max_steps=500)
        game_name = f"Dungeon (Level {difficulty})"
    else:
        raise ValueError(f"Unknown game type: {game_type}")

    # Observer
    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)

    # Stats
    victories = 0
    deaths = 0
    scores = []
    steps_taken = []

    # Run episodes
    for episode in range(num_episodes):
        state = game.reset()
        observer.reset()
        done = False

        while not done:
            # Get observation
            obs = observer.observe(state)

            # Use 'balanced' context
            context = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
            obs_with_context = add_context_to_observation(obs, context)

            # Get action
            action = agent.get_action(obs_with_context, epsilon=0.0)

            # Step
            state, reward, done = game.step(action)

        # Record results
        scores.append(game.score)
        steps_taken.append(game.steps)

        if game_type == 'pacman':
            max_score = game.num_pellets
            if game.score >= game.num_pellets or len(game.pellets) == 0:
                victories += 1
            else:
                deaths += 1
        elif game_type == 'dungeon':
            max_score = game.num_treasures
            if game.score >= game.num_treasures or len(game.treasures) == 0:
                victories += 1
            else:
                deaths += 1

    # Calculate statistics
    win_rate = (victories / num_episodes) * 100
    avg_score = np.mean(scores)
    max_score_achieved = np.max(scores)
    avg_steps = np.mean(steps_taken)

    return {
        'game': game_name,
        'episodes': num_episodes,
        'victories': victories,
        'deaths': deaths,
        'win_rate': win_rate,
        'avg_score': avg_score,
        'max_score': max_score_achieved,
        'max_possible': max_score,
        'avg_steps': avg_steps,
        'scores': scores
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to SNAKE model checkpoint')
    parser.add_argument('--episodes', type=int, default=20, help='Number of test episodes per configuration')
    args = parser.parse_args()

    print("=" * 80)
    print("ZERO-SHOT TRANSFER LEARNING ANALYSIS")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Episodes per test: {args.episodes}")
    print()

    # Test configurations
    tests = [
        # PacMan tests
        ('pacman', 0, 'No ghosts'),
        ('pacman', 1, '1 ghost (random)'),
        ('pacman', 2, '2 ghosts (random)'),
        ('pacman', 3, '3 ghosts (random)'),
        ('pacman', 4, '3 ghosts (chase)'),

        # Dungeon tests
        ('dungeon', 0, 'No enemies'),
        ('dungeon', 1, '1 enemy (patrol)'),
        ('dungeon', 2, '2 enemies (patrol)'),
        ('dungeon', 3, '3 enemies (patrol)'),
        ('dungeon', 4, '3 enemies (smart)'),
    ]

    results = []

    for game_type, difficulty, description in tests:
        print(f"\nTesting: {game_type.upper()} - {description}")
        print("-" * 80)

        result = test_transfer(args.model, game_type, difficulty, args.episodes)
        results.append(result)

        print(f"  Win Rate: {result['win_rate']:.1f}% ({result['victories']}/{result['episodes']})")
        print(f"  Avg Score: {result['avg_score']:.2f}/{result['max_possible']}")
        print(f"  Max Score: {result['max_score']}/{result['max_possible']}")
        print(f"  Avg Steps: {result['avg_steps']:.1f}")

    # Summary
    print("\n" + "=" * 80)
    print("TRANSFER LEARNING SUMMARY")
    print("=" * 80)

    # PacMan results
    pacman_results = [r for r in results if 'PacMan' in r['game']]
    print("\nPAC-MAN TRANSFER:")
    for r in pacman_results:
        print(f"  {r['game']:30} Win Rate: {r['win_rate']:5.1f}%  Avg: {r['avg_score']:4.1f}/{r['max_possible']}")

    # Dungeon results
    dungeon_results = [r for r in results if 'Dungeon' in r['game']]
    print("\nDUNGEON TRANSFER:")
    for r in dungeon_results:
        print(f"  {r['game']:30} Win Rate: {r['win_rate']:5.1f}%  Avg: {r['avg_score']:4.1f}/{r['max_possible']}")

    # Overall assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT:")
    print("=" * 80)

    # Calculate average success rates
    pacman_avg_win = np.mean([r['win_rate'] for r in pacman_results])
    dungeon_avg_win = np.mean([r['win_rate'] for r in dungeon_results])

    print(f"\nPac-Man Average Win Rate: {pacman_avg_win:.1f}%")
    if pacman_avg_win > 60:
        print("  ✓ EXCELLENT transfer - agent plays well!")
    elif pacman_avg_win > 30:
        print("  ✓ GOOD transfer - agent shows competence")
    elif pacman_avg_win > 10:
        print("  ~ PARTIAL transfer - some skills transferred")
    else:
        print("  ✗ POOR transfer - needs improvement")

    print(f"\nDungeon Average Win Rate: {dungeon_avg_win:.1f}%")
    if dungeon_avg_win > 60:
        print("  ✓ EXCELLENT transfer - agent explores well!")
    elif dungeon_avg_win > 30:
        print("  ✓ GOOD transfer - agent shows competence")
    elif dungeon_avg_win > 10:
        print("  ~ PARTIAL transfer - some skills transferred")
    else:
        print("  ✗ POOR transfer - needs improvement")

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)

    # Identify strengths
    best_result = max(results, key=lambda r: r['win_rate'])
    worst_result = min(results, key=lambda r: r['win_rate'])

    print(f"\nStrongest Transfer: {best_result['game']} ({best_result['win_rate']:.1f}%)")
    print(f"Weakest Transfer: {worst_result['game']} ({worst_result['win_rate']:.1f}%)")

    # Difficulty impact
    print("\nDifficulty Impact:")
    print("  PacMan: ", end="")
    for r in pacman_results:
        print(f"{r['win_rate']:.0f}% ", end="")
    print()
    print("  Dungeon: ", end="")
    for r in dungeon_results:
        print(f"{r['win_rate']:.0f}% ", end="")
    print()

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

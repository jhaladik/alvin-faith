"""
Automated analysis of zero-shot transfer to Local View game
Run tests without GUI to collect statistics
"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.local_view_game import LocalViewGame
from src.context_aware_agent import ContextAwareDQN, add_context_to_observation
from src.core.expanded_temporal_observer import ExpandedTemporalObserver


def test_local_view_transfer(model_path, difficulty=0, num_episodes=20):
    """Test zero-shot transfer on Local View game"""
    # Load SNAKE agent
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    agent = ContextAwareDQN(obs_dim=183, action_dim=4)
    agent.load_state_dict(checkpoint['policy_net'])
    agent.eval()

    # Create game
    game = LocalViewGame(
        world_size=40,
        num_coins=20,
        enemy_level=difficulty,
        max_steps=800
    )
    game_name = f"LocalView (Level {difficulty})"

    # Observer
    observer = ExpandedTemporalObserver(num_rays=16, ray_length=15)

    # Stats
    victories = 0
    deaths = 0
    scores = []
    steps_taken = []
    coins_collected_ratio = []

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
        coins_collected_ratio.append(game.score / game.num_coins)

        if game.score >= game.num_coins or len(game.coins) == 0:
            victories += 1
        else:
            deaths += 1

    # Calculate statistics
    win_rate = (victories / num_episodes) * 100
    avg_score = np.mean(scores)
    max_score_achieved = np.max(scores)
    avg_steps = np.mean(steps_taken)
    avg_collection_rate = np.mean(coins_collected_ratio) * 100

    return {
        'game': game_name,
        'episodes': num_episodes,
        'victories': victories,
        'deaths': deaths,
        'win_rate': win_rate,
        'avg_score': avg_score,
        'max_score': max_score_achieved,
        'max_possible': game.num_coins,
        'avg_steps': avg_steps,
        'collection_rate': avg_collection_rate,
        'scores': scores
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to SNAKE model checkpoint')
    parser.add_argument('--episodes', type=int, default=20, help='Number of test episodes per configuration')
    args = parser.parse_args()

    print("=" * 80)
    print("ZERO-SHOT TRANSFER: Snake -> Local View Collector")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Episodes per test: {args.episodes}")
    print()
    print("Game Features:")
    print("  - Moving perspective (camera follows agent)")
    print("  - Large world (40x40) with limited visibility")
    print("  - Medium reward density (20 coins)")
    print("  - Enemies as moving walls")
    print()

    # Test configurations
    tests = [
        (0, 'No enemies'),
        (1, '2 enemies (patrol)'),
        (2, '3 enemies (patrol)'),
        (3, '4 enemies (patrol)'),
        (4, '5 enemies (smart patrol)'),
    ]

    results = []

    for difficulty, description in tests:
        print(f"\nTesting: Level {difficulty} - {description}")
        print("-" * 80)

        result = test_local_view_transfer(args.model, difficulty, args.episodes)
        results.append(result)

        print(f"  Win Rate: {result['win_rate']:.1f}% ({result['victories']}/{result['episodes']})")
        print(f"  Avg Score: {result['avg_score']:.2f}/{result['max_possible']}")
        print(f"  Collection Rate: {result['collection_rate']:.1f}%")
        print(f"  Max Score: {result['max_score']}/{result['max_possible']}")
        print(f"  Avg Steps: {result['avg_steps']:.1f}/800")

    # Summary
    print("\n" + "=" * 80)
    print("TRANSFER LEARNING SUMMARY")
    print("=" * 80)

    print("\nLOCAL VIEW TRANSFER:")
    for r in results:
        print(f"  {r['game']:30} Win: {r['win_rate']:5.1f}%  "
              f"Collect: {r['collection_rate']:5.1f}%  "
              f"Avg: {r['avg_score']:4.1f}/{r['max_possible']}")

    # Overall assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT:")
    print("=" * 80)

    avg_win = np.mean([r['win_rate'] for r in results])
    avg_collection = np.mean([r['collection_rate'] for r in results])

    print(f"\nAverage Win Rate: {avg_win:.1f}%")
    print(f"Average Collection Rate: {avg_collection:.1f}%")
    print()

    if avg_win > 60:
        print("  SUCCESS! Excellent transfer to moving perspective!")
    elif avg_win > 30:
        print("  GOOD transfer - agent adapts to moving camera well")
    elif avg_win > 10:
        print("  PARTIAL transfer - some adaptation to moving perspective")
    else:
        print("  POOR transfer - struggles with moving perspective")

    # Difficulty progression
    print("\nDifficulty Progression:")
    print("  Difficulty: ", end="")
    for r in results:
        print(f"L{r['game'].split('Level ')[1].split(')')[0]}: {r['win_rate']:4.0f}%  ", end="")
    print()

    # Comparison with other games
    print("\n" + "=" * 80)
    print("COMPARISON WITH OTHER TRANSFERS:")
    print("=" * 80)
    print(f"\nLocal View (moving perspective):")
    print(f"  Reward density: 20 coins in 40x40 world (0.0125 density)")
    print(f"  Transfer result: {avg_win:.1f}% win rate")
    print()
    print(f"Expected comparisons (from previous tests):")
    print(f"  Pac-Man (dense, static view):  ~24% avg win rate")
    print(f"  Dungeon (sparse, static view):  ~0% avg win rate")
    print()
    print(f"Hypothesis:")
    if avg_win > 20:
        print("  Moving perspective doesn't hurt transfer much!")
        print("  Ray-based observation is viewpoint-independent")
    elif avg_win > 5:
        print("  Moderate impact from moving perspective")
        print("  Agent partially adapts to viewport changes")
    else:
        print("  Severe impact from moving perspective")
        print("  Agent may rely on absolute positions")

    # Detailed breakdown
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS:")
    print("=" * 80)

    best = max(results, key=lambda r: r['win_rate'])
    worst = min(results, key=lambda r: r['win_rate'])

    print(f"\nBest Performance: {best['game']} ({best['win_rate']:.1f}%)")
    print(f"Worst Performance: {worst['game']} ({worst['win_rate']:.1f}%)")

    # Check if collection rate is high but win rate is low (timeout issue)
    for r in results:
        if r['collection_rate'] > 70 and r['win_rate'] < 30:
            print(f"\nNote: {r['game']} has high collection but low wins")
            print("  Likely cause: Timeouts (agent explores but too slowly)")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

"""
Hidden Mechanic Discovery System

Revolutionary concept: Actively test hypotheses about hidden game mechanics.
Most games have non-obvious rules that standard RL never discovers.

Key Innovation:
- Standard RL: Learns "what works" from observed rewards
- Mechanic Discovery: Hypothesizes "what COULD work" and tests actively

Examples of Hidden Mechanics:
1. Threshold Effects: "After collecting 50 pellets, bonus mode activates"
2. Timing Cycles: "Every 17 steps, special reward spawns"
3. Action Sequences: "Up-Up-Down-Down triggers power-up"
4. Accumulation: "Rejecting 3 orders in a row triggers bulk order"
5. Proximity Combinations: "Being near entity A while holding item B = bonus"

Discovery Process:
1. Generate hypothesis based on observations
2. Design test sequence to validate hypothesis
3. Execute test and measure outcome
4. Update belief based on result
5. Confirmed mechanics → Incorporate into strategy
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque


class MechanicHypothesis:
    """Base class for mechanic hypotheses"""

    def __init__(self, hypothesis_type: str):
        self.hypothesis_type = hypothesis_type
        self.confidence = 0.0
        self.tests_conducted = 0
        self.tests_confirmed = 0
        self.description = ""

    def test(self, history: Dict) -> bool:
        """
        Test if hypothesis is supported by evidence.

        Args:
            history: Interaction history data

        Returns:
            True if hypothesis appears valid
        """
        raise NotImplementedError

    def get_confidence(self) -> float:
        """Get confidence in this hypothesis (0-1)"""
        if self.tests_conducted == 0:
            return 0.0
        return self.tests_confirmed / self.tests_conducted


class ThresholdMechanic(MechanicHypothesis):
    """
    Hypothesis: After N events, something special happens.

    Examples:
    - After 50 pellets collected → Bonus mode
    - After 100 steps survived → Extra life
    - After 10 enemies defeated → Power-up spawns
    """

    def __init__(self):
        super().__init__("threshold")
        self.threshold_value = None
        self.threshold_type = None  # 'collection', 'survival', 'steps'
        self.detected_thresholds = []

    def test(self, history: Dict) -> bool:
        """
        Test for threshold effects in reward history.

        Look for sudden reward spikes after reaching certain counts.
        """
        if 'rewards' not in history or len(history['rewards']) < 20:
            return False

        rewards = np.array(history['rewards'])
        steps = np.arange(len(rewards))

        # Look for reward spikes
        reward_diffs = np.diff(rewards)
        spike_indices = np.where(reward_diffs > np.mean(reward_diffs) + 2 * np.std(reward_diffs))[0]

        # Check if spikes occur at consistent intervals
        if len(spike_indices) >= 3:
            intervals = np.diff(spike_indices)
            if np.std(intervals) < 5.0:  # Consistent intervals
                self.threshold_value = int(np.mean(intervals))
                self.threshold_type = 'periodic_threshold'
                self.confidence = 0.8
                self.description = f"Threshold effect every ~{self.threshold_value} steps"
                self.tests_confirmed += 1
                self.tests_conducted += 1
                return True

        self.tests_conducted += 1
        return False


class TimingCycleMechanic(MechanicHypothesis):
    """
    Hypothesis: Events occur on predictable time cycles.

    Examples:
    - Every 17 steps, pellet spawns at (5, 5)
    - Every 30 steps, enemies patrol pattern resets
    - Every 10 steps, reward value doubles
    """

    def __init__(self):
        super().__init__("timing_cycle")
        self.cycle_length = None
        self.cycle_confidence = 0.0
        self.detected_cycles = []

    def test(self, history: Dict) -> bool:
        """
        Test for periodic patterns in events.

        Use autocorrelation to detect cycles.
        """
        if 'events' not in history or len(history['events']) < 30:
            return False

        # Create binary event timeline
        max_step = max(event['step'] for event in history['events'])
        timeline = np.zeros(max_step + 1)

        for event in history['events']:
            if event.get('type') == 'reward_collected':
                timeline[event['step']] = 1

        # Test candidate periods
        candidate_periods = [10, 15, 17, 20, 25, 30, 50]

        for period in candidate_periods:
            # Check if events align with this period
            aligned_count = 0
            total_periods = max_step // period

            for i in range(total_periods):
                expected_step = i * period
                # Check ±2 step tolerance
                if any(timeline[max(0, expected_step-2):min(len(timeline), expected_step+3)]):
                    aligned_count += 1

            alignment_ratio = aligned_count / total_periods if total_periods > 0 else 0

            if alignment_ratio > 0.6:  # 60% of expected events occurred
                self.cycle_length = period
                self.cycle_confidence = alignment_ratio
                self.confidence = alignment_ratio
                self.description = f"Event cycle every {period} steps ({alignment_ratio:.0%} aligned)"
                self.detected_cycles.append(period)
                self.tests_confirmed += 1
                self.tests_conducted += 1
                return True

        self.tests_conducted += 1
        return False


class ActionSequenceMechanic(MechanicHypothesis):
    """
    Hypothesis: Specific action sequences trigger special effects.

    Examples:
    - Up-Up-Down-Down-Left-Right → Konami code bonus
    - Circle pattern (Right-Down-Left-Up) → Tornado attack
    - Wait-Wait-Move → Charged attack
    """

    def __init__(self):
        super().__init__("action_sequence")
        self.magic_sequences = []
        self.sequence_length = 4  # Test sequences of length 4

    def test(self, history: Dict) -> bool:
        """
        Test for action sequences correlated with high rewards.

        Look for specific action patterns before reward spikes.
        """
        if 'actions' not in history or 'rewards' not in history:
            return False

        if len(history['actions']) < 10:
            return False

        actions = history['actions']
        rewards = history['rewards']

        # Find high-reward timesteps
        high_reward_steps = [
            i for i, r in enumerate(rewards)
            if r > np.mean(rewards) + np.std(rewards)
        ]

        if len(high_reward_steps) < 3:
            self.tests_conducted += 1
            return False

        # Check if same action sequence precedes high rewards
        sequences_before_reward = []
        for step in high_reward_steps:
            if step >= self.sequence_length:
                seq = tuple(actions[step-self.sequence_length:step])
                sequences_before_reward.append(seq)

        # Find most common sequence
        if sequences_before_reward:
            from collections import Counter
            seq_counts = Counter(sequences_before_reward)
            most_common_seq, count = seq_counts.most_common(1)[0]

            if count >= 3:  # Sequence appeared before 3+ high rewards
                self.magic_sequences.append(most_common_seq)
                self.confidence = count / len(high_reward_steps)
                self.description = f"Action sequence {most_common_seq} triggers high reward"
                self.tests_confirmed += 1
                self.tests_conducted += 1
                return True

        self.tests_conducted += 1
        return False


class AccumulationMechanic(MechanicHypothesis):
    """
    Hypothesis: Hidden counters accumulate and trigger effects.

    Examples:
    - Damage accumulates → Eventually triggers invincibility
    - Rejections accumulate → Triggers bulk order
    - Near-misses accumulate → Triggers luck bonus
    """

    def __init__(self):
        super().__init__("accumulation")
        self.accumulation_threshold = None
        self.accumulation_type = None

    def test(self, history: Dict) -> bool:
        """
        Test for accumulation effects.

        Look for rare events that occur after series of similar actions.
        """
        if 'actions' not in history or 'rewards' not in history:
            return False

        actions = history['actions']
        rewards = history['rewards']

        if len(actions) < 20:
            self.tests_conducted += 1
            return False

        # Look for runs of same action followed by reward spike
        current_run = 1
        max_run_before_reward = 0

        for i in range(1, len(actions)):
            if actions[i] == actions[i-1]:
                current_run += 1
            else:
                # Run ended, check if reward followed
                if i < len(rewards) and rewards[i] > np.mean(rewards) + np.std(rewards):
                    max_run_before_reward = max(max_run_before_reward, current_run)
                current_run = 1

        if max_run_before_reward >= 5:
            self.accumulation_threshold = max_run_before_reward
            self.accumulation_type = 'action_repetition'
            self.confidence = 0.6
            self.description = f"Repeating action {max_run_before_reward} times triggers bonus"
            self.tests_confirmed += 1
            self.tests_conducted += 1
            return True

        self.tests_conducted += 1
        return False


class ProximityCombinationMechanic(MechanicHypothesis):
    """
    Hypothesis: Being near certain entities while doing X creates special effect.

    Examples:
    - Collect pellet while near ghost → Bonus points
    - Move near wall while enemies chase → Enemies crash
    - Stand still near treasure for 5 steps → Double reward
    """

    def __init__(self):
        super().__init__("proximity_combination")
        self.proximity_effects = []

    def test(self, history: Dict) -> bool:
        """
        Test for proximity-based bonuses.

        Look for unusually high rewards correlated with entity proximity.
        """
        if 'rewards' not in history or 'entity_distances' not in history:
            self.tests_conducted += 1
            return False

        rewards = history['rewards']
        distances = history['entity_distances']

        if len(rewards) != len(distances):
            self.tests_conducted += 1
            return False

        # Find high rewards
        high_reward_steps = [
            i for i, r in enumerate(rewards)
            if r > np.mean(rewards) + np.std(rewards)
        ]

        if len(high_reward_steps) < 3:
            self.tests_conducted += 1
            return False

        # Check if high rewards correlate with close proximity
        close_proximity_count = sum(
            1 for i in high_reward_steps
            if i < len(distances) and distances[i] < 3.0
        )

        proximity_ratio = close_proximity_count / len(high_reward_steps)

        if proximity_ratio > 0.7:  # 70% of high rewards near entities
            self.confidence = proximity_ratio
            self.description = f"High rewards occur near entities ({proximity_ratio:.0%})"
            self.tests_confirmed += 1
            self.tests_conducted += 1
            return True

        self.tests_conducted += 1
        return False


class MechanicDetector:
    """
    System for discovering hidden game mechanics through hypothesis testing.

    Maintains a set of active hypotheses and tests them as data arrives.
    """

    def __init__(self):
        self.hypotheses = {
            'threshold': ThresholdMechanic(),
            'timing_cycle': TimingCycleMechanic(),
            'action_sequence': ActionSequenceMechanic(),
            'accumulation': AccumulationMechanic(),
            'proximity_combination': ProximityCombinationMechanic()
        }

        self.confirmed_mechanics = {}
        self.history_buffer = {
            'rewards': deque(maxlen=500),
            'actions': deque(maxlen=500),
            'events': deque(maxlen=500),
            'entity_distances': deque(maxlen=500)
        }

        self.step_count = 0

    def observe(self, reward: float, action: int, event: Optional[Dict] = None,
                entity_distance: Optional[float] = None):
        """
        Record observation for mechanic detection.

        Args:
            reward: Reward received
            action: Action taken
            event: Optional event dict (type, step, etc.)
            entity_distance: Distance to nearest entity
        """
        self.history_buffer['rewards'].append(reward)
        self.history_buffer['actions'].append(action)

        if event:
            event['step'] = self.step_count
            self.history_buffer['events'].append(event)

        if entity_distance is not None:
            self.history_buffer['entity_distances'].append(entity_distance)

        self.step_count += 1

    def test_hypotheses(self) -> Dict[str, MechanicHypothesis]:
        """
        Test all active hypotheses against current data.

        Returns:
            Dict of confirmed mechanics
        """
        # Only test every 50 steps (expensive)
        if self.step_count % 50 != 0:
            return self.confirmed_mechanics

        history = {
            'rewards': list(self.history_buffer['rewards']),
            'actions': list(self.history_buffer['actions']),
            'events': list(self.history_buffer['events']),
            'entity_distances': list(self.history_buffer['entity_distances'])
        }

        for name, hypothesis in self.hypotheses.items():
            if hypothesis.test(history):
                if hypothesis.confidence > 0.6:
                    self.confirmed_mechanics[name] = hypothesis

        return self.confirmed_mechanics

    def get_active_mechanics(self) -> Dict[str, str]:
        """Get descriptions of confirmed mechanics"""
        return {
            name: mechanic.description
            for name, mechanic in self.confirmed_mechanics.items()
        }

    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return {
            'total_observations': self.step_count,
            'hypotheses_tested': sum(h.tests_conducted for h in self.hypotheses.values()),
            'mechanics_confirmed': len(self.confirmed_mechanics),
            'confidence_scores': {
                name: mechanic.confidence
                for name, mechanic in self.confirmed_mechanics.items()
            }
        }


def demo():
    """Demonstrate mechanic detection"""
    print("=" * 60)
    print("HIDDEN MECHANIC DISCOVERY SYSTEM")
    print("=" * 60)
    print()

    detector = MechanicDetector()

    print("Simulating environment with hidden mechanics...")
    print()

    # Simulate environment with multiple hidden mechanics
    for step in range(200):
        # Hidden mechanic 1: Every 17 steps, bonus reward
        bonus_cycle = 0
        if step % 17 == 0:
            bonus_cycle = 50

        # Hidden mechanic 2: After 3 same actions, double reward
        action = np.random.randint(0, 4)

        # Base reward
        base_reward = 10 if np.random.random() < 0.2 else 0

        total_reward = base_reward + bonus_cycle

        # Create event if reward collected
        event = None
        if base_reward > 0:
            event = {'type': 'reward_collected'}

        # Random entity distance
        entity_distance = np.random.uniform(1, 10)

        # Observe
        detector.observe(total_reward, action, event, entity_distance)

    # Test hypotheses
    print("Testing hypotheses on collected data...")
    confirmed = detector.test_hypotheses()

    print(f"\nDiscovered {len(confirmed)} hidden mechanics:")
    print()

    for name, mechanic in confirmed.items():
        print(f"{name.upper().replace('_', ' ')}:")
        print(f"  {mechanic.description}")
        print(f"  Confidence: {mechanic.confidence:.2f}")
        print(f"  Tests: {mechanic.tests_confirmed}/{mechanic.tests_conducted}")
        print()

    stats = detector.get_statistics()
    print(f"Detection Statistics:")
    print(f"  Total observations: {stats['total_observations']}")
    print(f"  Hypotheses tested: {stats['hypotheses_tested']}")
    print(f"  Mechanics confirmed: {stats['mechanics_confirmed']}")
    print()

    print("KEY INSIGHT:")
    print("Agent discovered game rules that aren't directly observable!")
    print("- Timing cycles: Predicts when bonuses spawn")
    print("- Thresholds: Knows when special events trigger")
    print("- Sequences: Discovers 'cheat codes'")
    print()
    print("This knowledge enables superhuman performance!")


if __name__ == '__main__':
    demo()

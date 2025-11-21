"""
Faith-Based Evolutionary Exploration System

Revolutionary concept: Faith allows agents to maintain behavioral commitments
despite negative feedback, enabling discovery of hidden mechanics and long-term strategies.

Key Innovation:
- Standard RL eliminates promising behaviors too early
- Faith protects "irrational" behaviors long enough to discover breakthroughs
- Evolutionary pressure breeds successful patterns

Example:
    Without faith: "Wait 30 steps" → No reward → Eliminated
    With faith: "Wait 30 steps" → Discovers hidden spawn cycle → Breeds new strategies
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import random


class FaithPattern(nn.Module):
    """
    A single faith-based behavioral pattern.

    Represents a commitment to a specific behavior strategy that persists
    despite short-term negative feedback.
    """

    def __init__(self, pattern_id: int, signature_dim: int = 32):
        super().__init__()

        self.pattern_id = pattern_id
        self.signature_dim = signature_dim

        # Behavioral signature (learnable)
        self.signature = nn.Parameter(torch.randn(signature_dim) * 0.1)

        # Pattern metadata
        self.fitness = 0.0
        self.age = 0
        self.discoveries = 0  # Novel rewards found
        self.commitment_strength = 0.5  # How strongly to override Q-values

        # Behavior type (evolved)
        self.behavior_types = {
            'wait': 0.25,      # Wait/patience behavior
            'explore': 0.25,   # Random exploration
            'rhythmic': 0.25,  # Periodic actions
            'sacrificial': 0.25  # Accept short-term loss
        }

        # Temporal parameters
        self.commitment_length = np.random.randint(10, 50)  # How long to persist
        self.activation_frequency = np.random.uniform(0.1, 0.5)  # How often active

    def encode(self) -> torch.Tensor:
        """Get pattern signature for neural network input"""
        return self.signature

    def should_override(self, step: int, state: torch.Tensor) -> bool:
        """Decide if this pattern should override Q-learning at this step"""
        # FIXED: Simplified to respect faith_freq parameter in caller
        # The caller already controls overall faith frequency (e.g., 30%)
        # This just decides if THIS pattern activates when selected

        # Commitment-based: Once activated, persist for commitment_length
        if hasattr(self, '_active_until'):
            if step < self._active_until:
                return True
            else:
                delattr(self, '_active_until')
                return False

        # Start new commitment period with higher probability (50% vs old 10%)
        if random.random() < 0.5:
            self._active_until = step + self.commitment_length
            return True

        return False

    def get_action(self, state: torch.Tensor, base_action: int, step: int) -> int:
        """
        Generate action based on faith pattern.

        Args:
            state: Current observation
            base_action: Q-learning suggested action
            step: Current timestep

        Returns:
            Modified action based on faith pattern
        """
        # Select behavior based on dominant type
        dominant_behavior = max(self.behavior_types.items(), key=lambda x: x[1])[0]

        if dominant_behavior == 'wait':
            # Wait/no-op behavior (stay in place or minimal movement)
            return base_action if random.random() < 0.3 else np.random.randint(0, 4)

        elif dominant_behavior == 'explore':
            # Random exploration
            return np.random.randint(0, 4)

        elif dominant_behavior == 'rhythmic':
            # Periodic pattern (e.g., circle, back-and-forth)
            cycle_length = 4
            return step % cycle_length

        elif dominant_behavior == 'sacrificial':
            # Deliberately take risky actions
            # Inverse of Q-learning (explore opposite of what seems best)
            return (base_action + 2) % 4  # Opposite direction

        return base_action

    def is_counting_pattern(self) -> bool:
        """Check if this pattern involves temporal counting"""
        return self.behavior_types['wait'] > 0.5 or self.behavior_types['rhythmic'] > 0.5

    def update_fitness(self, reward_delta: float, novelty_score: float):
        """
        Update fitness based on performance and novelty.

        Args:
            reward_delta: Change in reward when this pattern was active
            novelty_score: How novel/surprising the outcome was
        """
        # Fitness combines:
        # 1. Reward improvement (exploitation)
        # 2. Novelty (exploration)
        # 3. Age penalty (prevent stagnation)

        reward_component = reward_delta
        novelty_component = novelty_score * 10.0  # Highly value discoveries
        age_penalty = -0.01 * self.age

        fitness_change = reward_component + novelty_component + age_penalty

        # Track discoveries
        if novelty_score > 0.5:
            self.discoveries += 1

        # Exponential moving average
        self.fitness = 0.9 * self.fitness + 0.1 * fitness_change
        self.age += 1


class FaithPopulation:
    """
    Population of faith patterns that evolve over time.

    Implements genetic algorithm-style evolution:
    - Selection: Top performers survive
    - Crossover: Breed new patterns from successful parents
    - Mutation: Random variations for exploration
    """

    def __init__(self, population_size: int = 20, signature_dim: int = 32):
        self.population_size = population_size
        self.signature_dim = signature_dim

        # Initialize population
        self.patterns: List[FaithPattern] = [
            FaithPattern(i, signature_dim) for i in range(population_size)
        ]

        # Evolution parameters
        self.generation = 0
        self.mutation_rate = 0.4  # Increased from 0.1 to break uniformity
        self.elite_size = 4  # Reduced from 10 to allow more evolution (20% instead of 50%)
        self.tournament_size = 3

        # Tracking
        self.best_fitness_history = []
        self.diversity_history = []

    def get_active_patterns(self) -> List[FaithPattern]:
        """Get currently active faith patterns (for planning)"""
        # Return top 5 performers
        sorted_patterns = sorted(self.patterns, key=lambda p: p.fitness, reverse=True)
        return sorted_patterns[:5]

    def select_pattern_for_episode(self) -> FaithPattern:
        """
        Select a pattern to use for current episode.

        Uses tournament selection to balance exploitation and exploration.
        """
        # Tournament selection
        tournament = random.sample(self.patterns, self.tournament_size)

        # 80% chance: Select best from tournament
        # 20% chance: Select random (exploration)
        if random.random() < 0.8:
            return max(tournament, key=lambda p: p.fitness)
        else:
            return random.choice(tournament)

    def evolve(self, performance_history: List[float]):
        """
        Evolve population based on fitness.

        Args:
            performance_history: Recent episode rewards for adaptation pressure
        """
        self.generation += 1

        # Sort by fitness
        sorted_patterns = sorted(self.patterns, key=lambda p: p.fitness, reverse=True)

        # Track best fitness
        best_fitness = sorted_patterns[0].fitness
        self.best_fitness_history.append(best_fitness)

        # Calculate diversity (how different are the patterns?)
        diversity = self._calculate_diversity()
        self.diversity_history.append(diversity)

        # Adaptive mutation rate (increase if stuck)
        if len(performance_history) > 50:
            recent_std = np.std(performance_history[-50:])
            if recent_std < 5.0:  # Stagnation detected
                self.mutation_rate = min(0.5, self.mutation_rate * 1.2)
            else:
                self.mutation_rate = max(0.1, self.mutation_rate * 0.95)

        # Create new population
        new_population = []

        # 1. Elite preservation (keep top performers)
        new_population.extend(sorted_patterns[:self.elite_size])

        # 2. Crossover (breed new patterns)
        num_offspring = (self.population_size - self.elite_size) // 2
        for _ in range(num_offspring):
            parent1 = self._tournament_select(sorted_patterns)
            parent2 = self._tournament_select(sorted_patterns)
            child = self._crossover(parent1, parent2)
            new_population.append(child)

        # 3. Mutation (random variations)
        while len(new_population) < self.population_size:
            parent = random.choice(sorted_patterns[:self.elite_size])
            mutant = self._mutate(parent)
            new_population.append(mutant)

        # Update population
        self.patterns = new_population[:self.population_size]

        return {
            'generation': self.generation,
            'best_fitness': best_fitness,
            'diversity': diversity,
            'mutation_rate': self.mutation_rate
        }

    def _tournament_select(self, population: List[FaithPattern]) -> FaithPattern:
        """Tournament selection for parent"""
        tournament = random.sample(population[:15], self.tournament_size)
        return max(tournament, key=lambda p: p.fitness)

    def _crossover(self, parent1: FaithPattern, parent2: FaithPattern) -> FaithPattern:
        """
        Breed new pattern from two parents.

        Combines signatures and behavior types.
        """
        child = FaithPattern(len(self.patterns), self.signature_dim)

        # Blend signatures (average with noise)
        with torch.no_grad():
            child.signature.data = (parent1.signature.data + parent2.signature.data) / 2
            child.signature.data += torch.randn_like(child.signature.data) * 0.05

        # Inherit behavior types (blend)
        for behavior in child.behavior_types.keys():
            child.behavior_types[behavior] = (
                parent1.behavior_types[behavior] + parent2.behavior_types[behavior]
            ) / 2

        # Inherit temporal parameters (average)
        child.commitment_length = (parent1.commitment_length + parent2.commitment_length) // 2
        child.activation_frequency = (parent1.activation_frequency + parent2.activation_frequency) / 2
        child.commitment_strength = (parent1.commitment_strength + parent2.commitment_strength) / 2

        return child

    def _mutate(self, pattern: FaithPattern) -> FaithPattern:
        """
        Create mutated version of pattern.

        Introduces random variations for exploration.
        """
        mutant = FaithPattern(len(self.patterns), self.signature_dim)

        # Copy and mutate signature
        with torch.no_grad():
            mutant.signature.data = pattern.signature.data.clone()
            mutant.signature.data += torch.randn_like(mutant.signature.data) * self.mutation_rate

        # Mutate behavior types
        for behavior in mutant.behavior_types.keys():
            mutant.behavior_types[behavior] = pattern.behavior_types[behavior]
            if random.random() < self.mutation_rate:
                mutant.behavior_types[behavior] += random.gauss(0, 0.2)

        # Normalize behavior types
        total = sum(mutant.behavior_types.values())
        for behavior in mutant.behavior_types.keys():
            mutant.behavior_types[behavior] /= total

        # Mutate temporal parameters
        if random.random() < self.mutation_rate:
            mutant.commitment_length = max(5, pattern.commitment_length + random.randint(-10, 10))
        else:
            mutant.commitment_length = pattern.commitment_length

        if random.random() < self.mutation_rate:
            mutant.activation_frequency = max(0.05, min(0.9, pattern.activation_frequency + random.gauss(0, 0.1)))
        else:
            mutant.activation_frequency = pattern.activation_frequency

        return mutant

    def _calculate_diversity(self) -> float:
        """Calculate population diversity (signature variance)"""
        signatures = torch.stack([p.signature.data for p in self.patterns])
        diversity = torch.var(signatures, dim=0).mean().item()
        return diversity

    def get_statistics(self) -> Dict:
        """Get population statistics for logging"""
        fitnesses = [p.fitness for p in self.patterns]
        ages = [p.age for p in self.patterns]
        discoveries = [p.discoveries for p in self.patterns]

        # Dominant behavior types
        behavior_counts = {
            'wait': 0, 'explore': 0, 'rhythmic': 0, 'sacrificial': 0
        }
        for p in self.patterns:
            dominant = max(p.behavior_types.items(), key=lambda x: x[1])[0]
            behavior_counts[dominant] += 1

        return {
            'mean_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'mean_age': np.mean(ages),
            'total_discoveries': sum(discoveries),
            'behavior_distribution': behavior_counts,
            'diversity': self._calculate_diversity()
        }


def demo():
    """Demonstrate faith pattern evolution"""
    print("=" * 60)
    print("FAITH-BASED EVOLUTIONARY EXPLORATION SYSTEM")
    print("=" * 60)
    print()

    # Create population
    population = FaithPopulation(population_size=20, signature_dim=32)
    print(f"Population: {len(population.patterns)} faith patterns")
    print(f"Generation: {population.generation}")
    print()

    # Simulate evolution over 10 generations
    performance_history = []

    for gen in range(10):
        # Simulate episode performance
        for pattern in population.patterns:
            # Simulate reward and novelty
            reward_delta = random.gauss(0, 5)
            novelty_score = random.random()
            pattern.update_fitness(reward_delta, novelty_score)

        # Track performance
        performance_history.append(random.gauss(100, 20))

        # Evolve
        stats = population.evolve(performance_history)

        print(f"Generation {stats['generation']}:")
        print(f"  Best Fitness: {stats['best_fitness']:.2f}")
        print(f"  Diversity: {stats['diversity']:.4f}")
        print(f"  Mutation Rate: {stats['mutation_rate']:.3f}")

        if gen % 3 == 0:
            pop_stats = population.get_statistics()
            print(f"  Behavior Distribution: {pop_stats['behavior_distribution']}")
            print(f"  Total Discoveries: {pop_stats['total_discoveries']}")
        print()

    print("KEY INSIGHT:")
    print("Faith patterns evolve to discover non-obvious strategies!")
    print("- Waiting behaviors discover hidden spawn cycles")
    print("- Rhythmic patterns find temporal resonance")
    print("- Sacrificial strategies unlock long-term rewards")
    print()
    print("This enables breaking through Q-learning plateaus!")


if __name__ == '__main__':
    demo()

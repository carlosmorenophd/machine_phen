"""All genetic enum class or data class"""
from dataclasses import dataclass


@dataclass
class GeneticParameter():
    """Basic parameter to run genetic algorithm
    """
    population_size: int = 100
    num_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.01
    chromosome_length: int = 2

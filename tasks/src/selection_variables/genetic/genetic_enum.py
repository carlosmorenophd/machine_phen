"""All genetic enum class or data class"""
from dataclasses import dataclass

from src.machines.machine_data_frame_handler import DataFrameHandler
from src.machines.regression_run import MachineRegression
from src.selection_variables.selection_data_frame_handler import SelectionBestMetric
from src.machines.machine_enums import MachineJson



@dataclass
class GeneticParameter():
    """Basic parameter to run genetic algorithm
    """
    population_size: int = 100
    num_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.01
    chromosome_length: int = 2

@dataclass
class MachineMainRegression():
    """Basic parameters for machine regression models and dataset
    """
    data_frame: DataFrameHandler
    best_metric: SelectionBestMetric
    machine_definition: MachineJson
    machine: MachineRegression

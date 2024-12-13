"""All genetic enum class or data class"""
from dataclasses import dataclass
import json


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


def convert_str_genetic_parameters(genetic_parameters_str: str) -> GeneticParameter:
    """convert str from GeneticParameters to class

    Args:
        genetic_parameters_str (str): str to cast

    Returns:
        GeneticParameter: object with default parameters or setting by str
    """
    parameters = GeneticParameter()
    parameters_json = json.loads(genetic_parameters_str)
    if "population_size" in parameters_json:
        parameters.population_size = parameters_json["population_size"]
    if "num_generations" in parameters_json:
        parameters.num_generations = parameters_json["num_generations"]
    if "crossover_rate" in parameters_json:
        parameters.crossover_rate = parameters_json["crossover_rate"]
    if "mutation_rate" in parameters_json:
        parameters.mutation_rate = parameters_json["mutation_rate"]
    if "chromosome_length" in parameters_json:
        parameters.chromosome_length = parameters_json["chromosome_length"]
    return parameters

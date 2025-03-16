"""Operative class to run the genetic algorithm.
"""

from src.optimizations.optimization_enum import (
    GeneticAlgorithmParameter,
)
from src.files.file_machine import FileDataRegression
from src.optimizations.optimization_genetic_algorithm import GeneticAlgorithm
from src.helpers.file_access import FileData
from src.optimizations.optimization_features import FeatureForwardBackwardSelection


def optimization_run_from_task(
    file_data: FileDataRegression,
) -> None:
    """Launch the genetic algorithm
    """
    genetic_algorithm_parameters = GeneticAlgorithmParameter()
    genetic_algorithm = GeneticAlgorithm(
        file_data=file_data,
        training_data=genetic_algorithm_parameters.training_data,
        genetic_algorithm_parameters=genetic_algorithm_parameters,
    )
    genetic_algorithm.run()
    genetic_algorithm.export()


def forward_backward_features(
    file_data: FileDataRegression,
    file_models: FileData,
) -> None:
    """Run the backward feature selection and whe it finish run the forward feature selection
    """
    features = FeatureForwardBackwardSelection(
        file_data=file_data,
        file_models=file_models,
    )
    features.run()

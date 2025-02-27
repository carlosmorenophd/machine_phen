"""Operative class to run the genetic algorithm.
"""

from src.optimizations.optimization_enum import (
    GeneticAlgorithmParameter,
    SearchMode,
    search_mode_from_search_mode,
)
from src.files.file_machine import FileDataRegression
from src.optimizations.optimization_genetic_algorithm import GeneticAlgorithm
from src.helpers.file_access import FileData
from src.optimizations.optimization_features import FeatureForwardBackwardSelection


def optimization_run_from_task(
    file_data: FileDataRegression,
    search_mode: SearchMode,
) -> None:
    """Launch the genetic algorithm
    """
    training_info = search_mode_from_search_mode(
        search_mode=search_mode
    )
    genetic_algorithm_parameters = GeneticAlgorithmParameter(
        search_mode=search_mode)
    genetic_algorithm = GeneticAlgorithm(
        file_data=file_data,
        training_info=training_info,
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

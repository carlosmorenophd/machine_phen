"""Operative class to run the genetic algorithm.
"""

from src.optimizations.optimization_enum import (
    GeneticAlgorithmParameter,
    FileProcessFeatureSelection,
)
from src.files.file_machine import FileDataRegression
from src.optimizations.optimization_genetic_algorithm import GeneticAlgorithm
from src.optimizations.optimization_features import (
    FeatureSelection,
)


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
    file_name: str,
    file_json_definition: str,
) -> None:
    """Run the backward feature selection and whe it finish
        run the forward feature selection
    """
    file_process = FileProcessFeatureSelection(
        file_json_definition=file_json_definition,
        file_name=file_name
    )

    features = FeatureSelection(
        file_process=file_process
    )
    features.run()

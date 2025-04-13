"""Operative class to run the genetic algorithm.
"""

import re
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
    sort_columns: str,
) -> None:
    """Launch the genetic algorithm
    """
    if sort_columns != "" or sort_columns is not None:
        file_data.re_sort_columns(new_sort_columns=[
            re.sub(r'^[ \n\r\t]+', '', element)
            for element in sort_columns.split(',')
        ])

    genetic_algorithm_parameters = GeneticAlgorithmParameter()
    genetic_algorithm = GeneticAlgorithm(
        file_data=file_data,
        training_data=genetic_algorithm_parameters.training_data,
        genetic_algorithm_parameters=genetic_algorithm_parameters,
    )
    genetic_algorithm.run()
    genetic_algorithm.export()


def feature_selection_run(
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
    features.export()

""" All basic concepts and enums for work with genetics values"""
from enum import Enum
from abc import ABC
from dataclasses import dataclass

from src.machines.machine_enums import MachineNames
from src.files.file_machine import TrainingData


class SearchMode(Enum):
    """Type of search, define the number of population, the generation and the mutation rate
    """

    QUICK_EXPLORATION = "quick_exploration"
    BASIC_SEARCH = "basic_search"
    DEEP_SEARCH = "deep_search"


def convert_str_to_search_mode(search_mode_str: str) -> SearchMode:
    """Convert str to a valid search mode for genetic algorithm

    Args:
        search_mode_str (str): search mode

    Raises:
        ValueError: value not valid

    Returns:
        SearchMode: Search mode from enum
    """
    try:
        return SearchMode(search_mode_str)
    except ValueError as e:
        raise ValueError(
            f"Parameter '{search_mode_str}' is not a valid SearchMode") from e


def search_mode_from_search_mode(search_mode: SearchMode) -> TrainingData:
    """Return the training information from search mode

    Args:
        search_mode (SearchMode): search mode

    Returns:
        TrainingData: training information
    """
    if search_mode == SearchMode.QUICK_EXPLORATION:
        return TrainingData(
            test_size=0.8,
            random_state=42
        )
    if search_mode == SearchMode.BASIC_SEARCH:
        return TrainingData(
            test_size=0.7,
            random_state=42
        )
    if search_mode == SearchMode.DEEP_SEARCH:
        return TrainingData(
            test_size=0.6,
            random_state=42
        )
    raise ValueError(f"Search mode '{search_mode}' is not valid")


@dataclass
class GeneticAlgorithmParameter(ABC):
    """Configuration to run the genetic algorithm
    """

    def __init__(self, search_mode: SearchMode) -> None:
        self._search_mode = search_mode
        self._machines_key = []
        if search_mode == SearchMode.QUICK_EXPLORATION:
            # self._machines_key.append(MachineNames.RANDOM_FOREST_REGRESSION)
            self._machines_key.append(MachineNames.BAYESIAN_REGRESSION)
            self._number_population = 10
            self._number_generation = 20
            self._mutation_rate = 0.05
            self._cross_over_rate = 0.8
            self._hyper_parameter_deep_decimal = 3
            self._mutation_machine = 0.1
        if search_mode == SearchMode.BASIC_SEARCH:
            self._machines_key.append(MachineNames.BAYESIAN_REGRESSION)
            self._machines_key.append(
                MachineNames.EXTREME_GRADIENT_BOOSTING_REGRESSION)
            self._number_population = 50
            self._number_generation = 30
            self._mutation_rate = 0.1
            self._cross_over_rate = 0.7
            self._hyper_parameter_deep_decimal = 6
            self._mutation_machine = 0.2
        if search_mode == SearchMode.DEEP_SEARCH:
            self._machines_key.append(MachineNames.BAYESIAN_REGRESSION)
            self._machines_key.append(
                MachineNames.EXTREME_GRADIENT_BOOSTING_REGRESSION)
            self._machines_key.append(MachineNames.RANDOM_FOREST_REGRESSION)
            self._machines_key.append(MachineNames.SUPPORT_VECTOR_REGRESSION)
            self._number_population = 300
            self._number_generation = 50
            self._mutation_rate = 0.15
            self._cross_over_rate = 0.6
            self._hyper_parameter_deep_decimal = 9
            self._mutation_machine = 0.3

    @property
    def machines_key(self) -> list[MachineNames]:
        """Return the list of machines key

        Returns:
            list[MachineNames]: machines key
        """
        return self._machines_key

    @property
    def search_mode(self) -> SearchMode:
        """Return the search mode

        Returns:
            SearchMode: search mode
        """
        return self._search_mode

    @property
    def number_population(self) -> int:
        """Return the population

        Returns:
            int: population
        """

        return self._number_population

    @property
    def number_generation(self) -> int:
        """Return the number of generation


        Returns:
            int: number of generation
        """
        return self._number_generation

    @property
    def mutation_rate(self) -> float:
        """Return the mutation rate

        Returns:
            float: mutation rate
        """
        return self._mutation_rate

    @property
    def cross_over_rate(self) -> float:
        """Return the cross over rate

        Returns:
            float: cross over rate
        """
        return self._cross_over_rate

    @property
    def hyper_parameter_deep_decimal(self) -> int:
        """Return the deep decimal

        Returns:
            int: deep decimal
        """
        return self._hyper_parameter_deep_decimal

    @property
    def mutate_machine(self) -> float:
        """Return the mutate machine"""
        return self._mutation_machine


def convert_parameters_str_to_optimization(search_mode_str: str) -> GeneticAlgorithmParameter:
    """Convert str to a valid search mode for genetic algorithm

    Args:
        parameters_str (str): search mode

    Raises:
        ValueError: value not valid

    Returns:
        GeneticParameter: Search mode from enum
    """
    try:
        return SearchMode(search_mode_str)
    except ValueError as e:
        raise ValueError(
            f"Parameter '{search_mode_str}' is not a valid SearchMode") from e

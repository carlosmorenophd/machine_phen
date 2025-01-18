""" All basic concepts and enums for work with genetics values"""
from enum import Enum
from abc import ABC
from dataclasses import dataclass

from src.machines.machine_enums import MachineNames


class SearchMode(Enum):
    """Type of search, define the number of population, the generation and the mutation rate
    """

    QUICK_EXPLORATION = "quick_exploration"
    BASIC_SEARCH = "basic_search"
    DEEP_SEARCH = "deep_search"


@dataclass
class GeneticParameter(ABC):
    """Configuration to run the genetic algorithm
    """

    def __init__(self, search_mode: SearchMode) -> None:
        self._search_mode = search_mode
        self._machines_key = []
        if search_mode == SearchMode.QUICK_EXPLORATION:
            self._machines_key.append(MachineNames.BayesianRegression)
            self._population = 10
            self._generation = 10
            self._mutation_rate = 0.1
        elif search_mode == SearchMode.BASIC_SEARCH:
            self._machines_key.append(MachineNames.BayesianRegression)
            self._machines_key.append(MachineNames.ExtremeGradientBoostingRegression)
            self._population = 50
            self._generation = 50
            self._mutation_rate = 0.05
        else:
            self._machines_key.append(MachineNames.BayesianRegression)
            self._machines_key.append(MachineNames.ExtremeGradientBoostingRegression)
            self._machines_key.append(MachineNames.RandomForestRegression)
            self._machines_key.append(MachineNames.LASSORegression)
            self._machines_key.append(MachineNames.SupportVectorRegression)
            self._population = 100
            self._generation = 50
            self._mutation_rate = 0.01

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
    def population(self) -> int:
        """Return the population

        Returns:
            int: population
        """

        return self._population

    @property
    def generation(self) -> int:
        """Return the number of generation


        Returns:
            int: number of generation
        """
        return self._generation

    @property
    def mutation_rate(self) -> float:
        """Return the mutation rate

        Returns:
            float: mutation rate
        """
        return self._mutation_rate


def convert_parameters_str_to_genetic(search_mode_str: str) -> GeneticParameter:
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

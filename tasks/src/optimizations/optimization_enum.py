""" All basic concepts and enums for work with genetics values"""
from typing import List

from abc import ABC
from dataclasses import dataclass

from src.machines.machine_enums import MachineNames
from src.files.file_machine import TrainingData
from src.metrics.metric_enums import MetricEnum


class GeneticIndividualParameter():
    """Parameter to create a individual for genetic algorithm
    """

    def __init__(
        self,
        number_population: int = 100,
        hyper_parameter_deep_decimal: int = 5,
        metric_selection: MetricEnum = MetricEnum.ACCURACY_MEAN_ABSOLUTE_PERCENTAGE_ERROR,
        machines_key: List[MachineNames] = None,
    ) -> None:
        self._number_population = number_population
        self._hyper_parameter_deep_decimal = hyper_parameter_deep_decimal
        self._metric_selection = metric_selection
        self._machines_key = [
            MachineNames.LASSO_REGRESSION,
            MachineNames.EXTREME_GRADIENT_BOOSTING_REGRESSION,
            MachineNames.RANDOM_FOREST_REGRESSION,
            MachineNames.SUPPORT_VECTOR_REGRESSION,
        ]
        if machines_key is not None:
            self._machines_key = machines_key

    @property
    def number_population(self) -> int:
        """Return the number of population"""
        return self._number_population

    @property
    def hyper_parameter_deep_decimal(self) -> int:
        """Return the deep decimal"""
        return self._hyper_parameter_deep_decimal

    @property
    def metric_selection(self) -> MetricEnum:
        """Return the metric selection"""
        return self._metric_selection

    @property
    def machines_key(self) -> List[MachineNames]:
        """Return the list of machines key"""
        return self._machines_key


@dataclass
class GeneticMutationParameter():
    """Parameter to create a mutation for genetic algorithm
    """
    mutation_rate: float = 0.05
    cross_over_rate: float = 0.8
    mutate_machine: float = 0.1


@dataclass
class GeneticAlgorithmParameter(ABC):
    """Configuration to run the genetic algorithm
    """

    def __init__(self) -> None:
        self._number_generation: int = 100
        self._individual_parameter = GeneticIndividualParameter(
            hyper_parameter_deep_decimal=5,
            machines_key=[
                MachineNames.EXTREME_GRADIENT_BOOSTING_REGRESSION,
            ],
            metric_selection=MetricEnum.ACCURACY_MEAN_ABSOLUTE_PERCENTAGE_ERROR,
            number_population=120,
        )
        # self._number_generation: int = 120
        # self._individual_parameter = GeneticIndividualParameter()
        self._mutation_parameters = [
            GeneticMutationParameter(
                mutate_machine=0.8,
                cross_over_rate=0.8,
                mutation_rate=0.8,
            ),
            GeneticMutationParameter(
                mutate_machine=0.6,
                cross_over_rate=0.6,
                mutation_rate=0.6,
            ),
            GeneticMutationParameter(
                mutate_machine=0.2,
                cross_over_rate=0.3,
                mutation_rate=0.2,
            ),
        ]
        self._training_data = TrainingData()

    @property
    def machines_key(self) -> list[MachineNames]:
        """Return the list of machines key

        Returns:
            list[MachineNames]: machines key
        """
        return self._individual_parameter.machines_key

    @property
    def number_population(self) -> int:
        """Return the population

        Returns:
            int: population
        """

        return self._individual_parameter.number_population

    @property
    def number_generation(self) -> int:
        """Return the number of generation


        Returns:
            int: number of generation
        """
        return self._number_generation

    def get_mutation_rate(self, number_generation: int) -> float:
        """Return the mutation rate

        Returns:
            float: mutation rate
        """
        return self._get_parameters_by_number_generation(
            number_generation=number_generation
        ).mutation_rate

    def _get_parameters_by_number_generation(
            self,
            number_generation: int,
    ) -> GeneticMutationParameter:
        """Return the mutation parameters

        Returns:
            GeneticMutationParameter: mutation parameters
        """
        if self._number_generation * .40 < number_generation:
            return self._mutation_parameters[0]
        if self._number_generation * .80 < number_generation:
            return self._mutation_parameters[1]
        return self._mutation_parameters[2]

    def get_cross_over_rate(
            self, number_generation: int,
    ) -> float:
        """Return the cross over rate

        Returns:
            float: cross over rate
        """
        return self._get_parameters_by_number_generation(
            number_generation=number_generation
        ).cross_over_rate

    @property
    def hyper_parameter_deep_decimal(self) -> int:
        """Return the deep decimal

        Returns:
            int: deep decimal
        """
        return self._individual_parameter.hyper_parameter_deep_decimal

    def get_mutate_machine(self, number_generation: int) -> float:
        """Return the mutate machine"""
        return self._get_parameters_by_number_generation(
            number_generation=number_generation
        ).mutate_machine

    @property
    def metric_selection(self) -> MetricEnum:
        """Return the metric selection

        Returns:
            MetricEnum: metric selection
        """
        return self._individual_parameter.metric_selection

    @property
    def training_data(self) -> TrainingData:
        """Return the training data

        Returns:
            TrainingData: training data
        """
        return self._training_data

"""Operative class to run the genetic algorithm.
"""
import random

import pandas as pd

from src.optimizations.optimization_enum import (
    GeneticAlgorithmParameter,
    SearchMode,
    search_mode_from_search_mode,
)
from src.files.file_machine import FileData, TrainingData, FileMachine, DatasetOptimizationData
from src.machines.machine_build import machine_build_regression_optimization
from src.machines.machine import MachineRegression
from src.metrics.metric import Metric
from src.metrics.metric_enums import MetricEnum

# TODO: Store result on database
# TODO: Create function to extract result according to some rule, number of individual
# TODO: Review the generation of hyper parameters
# TODO: Create new random form hyper parameters to get values from .000001, .00001 and .0001
# TODO: Adding new cross over when are different machines
# TODO: Adding to chromosome features new cross over keep the most hight value
# TODO: Adding increase the rate in last values of individual


class GeneticIndividual():
    """Class to create a individual for genetic algorithm
    """

    def __init__(self, machine: MachineRegression = None) -> None:
        self._dataset: DatasetOptimizationData = None
        self._machine = None
        if machine is not None:
            self._machine = machine
        self._features_chromosome:  list[bool] = None
        self._metric: Metric = None
        self._index_metric: float = None

    def __str__(self) -> str:
        return f"machine: {self._machine}  features: {self._dataset.features_name}"

    @property
    def dataset(self):
        """Get the dataset"""
        return self._dataset

    @property
    def machine(self) -> MachineRegression:
        """Get the machine"""
        if self._machine is None:
            raise ValueError("Machine is not defined")
        return self._machine

    @property
    def index_metric(self):
        """Get the index metric"""
        return self._index_metric

    @property
    def features_chromosome(self):
        """Get the features chromosome"""
        return self._features_chromosome

    def run(self):
        """Run the machine with the features selected"""
        self._machine.build_machine()
        self._machine.training(x_train=self._dataset.x_train,
                               y_train=self._dataset.y_train)
        self._metric = self._machine.test(
            x_test=self._dataset.x_test, y_test=self._dataset.y_test)
        self._index_metric = self._metric.get_single_metric(
            metric=MetricEnum.ACCURACY_MEAN_ABSOLUTE_PERCENTAGE_ERROR)

    def set_features_chromosome(self, features_chromosome: list[bool]):
        """Set the features chromosome"""
        self._features_chromosome = features_chromosome

    def set_machine(self, machine: MachineRegression):
        """Set the machine"""
        self._machine = machine

    def apply_dataset(self, file_machine: FileData, features_chromosome: list[bool] = None):
        """Convert the features chromosome to dataset

        Args:
            file_machine (FileData): file with all information from dataset
            features_chromosome (list[bool], optional):
                Features chromosome to define the column to keep of dataset  Defaults to None.

        Raises:
            ValueError: Error if not was define features chromosome.
        """
        if features_chromosome is not None:
            self._features_chromosome = features_chromosome
        if self._features_chromosome is None:
            raise ValueError("Features chromosome is not defined")
        self._dataset = file_machine.get_dataset(
            columns_to_keep=[random.choice(
                [True, False]) for _ in self._features_chromosome]
        )

    def mutate_features_chromosome(self, mutation_rate: float):
        """Mutate the features chromosome

        Args:
            rate_mutation (float): Range between 0 and 1 to mutate the chromosome
        """
        self._features_chromosome = [
            not feature if random.random(
            ) < mutation_rate else feature
            for feature in self._features_chromosome
        ]

    def to_dictionary(self):
        """Convert the individual to dictionary

        Returns:
            dict: Dictionary with the individual information
        """
        general_dict = {
            "machine_name": self._machine.machine_name,
            "index_metric": self._index_metric,
        }
        general_dict.update(self._metric.get_all_metric())
        for key, _ in self._machine.hyper_parameters:
            general_dict[f"{self._machine.machine_name}_hyper_{
                key}"] = self._machine.hyper_parameters[key]
        for chromosome, feature in zip(self._features_chromosome, self._dataset.features_name):
            general_dict[f"feature_{feature}"] = chromosome
        return general_dict


class GeneticAlgorithm():
    """Main class to run the genetic algorithm for optimization process.
    """

    def __init__(
        self,
        file_data: FileData,
        training_info: TrainingData,
        genetic_algorithm_parameters:  GeneticAlgorithmParameter,
    ) -> None:
        self._file_machine = FileMachine(
            file_data=file_data,
            training_info=training_info,
        )
        self._genetic_algorithm_parameters = genetic_algorithm_parameters
        self._population: list[GeneticIndividual] = []
        self._new_population: list[GeneticIndividual] = []
        self._global_population: list[GeneticIndividual] = []

    def create_initial_population(self, population_number: int):
        """Create the initial population
        """
        for _ in range(population_number):
            self._population.append(
                self.create_initial_individual(
                    features=self._file_machine.columns_name_with_out_target
                )
            )

    def create_initial_individual(self, features: list[str]) -> GeneticIndividual:
        """Create a individual with the features selected

        Args:
            features (list[str]): list of columns of dataset

        Returns:
            GeneticIndividual: Individual with the features selected and one machine by random
        """
        genetic_individual = GeneticIndividual(
            machine=machine_build_regression_optimization(
                machine_name=random.choice(
                    self._genetic_algorithm_parameters.machines_key
                )
            )
        )
        genetic_individual.apply_dataset(
            file_machine=self._file_machine,
            features_chromosome=[random.choice(
                [True, False]) for _ in features]
        )
        return genetic_individual

    def crossover(self, parent_1: GeneticIndividual, parent_2: GeneticIndividual):
        """Crossover between two parents
        """
        crossover_point = random.randint(
            1, len(self._file_machine.columns_name_with_out_target) - 1)
        child_1 = GeneticIndividual()
        child_2 = GeneticIndividual()
        child_1.set_features_chromosome(parent_1.features_chromosome[
            :crossover_point] + parent_2.features_chromosome[crossover_point:])

        child_2.set_features_chromosome(parent_2.features_chromosome[
            :crossover_point] + parent_1.features_chromosome[crossover_point:])
        # Apply mutation
        hyper_parameters_1 = parent_1.machine.hyper_parameters
        hyper_parameters_2 = parent_2.machine.hyper_parameters
        if parent_1.machine.machine_name == parent_2.machine.machine_name:
            for key, _ in parent_1.machine.hyper_parameters.items():
                if random.random() < self._genetic_algorithm_parameters.cross_over_rate:
                    hyper_parameters_1[key].value = hyper_parameters_1[key].value
                    hyper_parameters_2[key].value = hyper_parameters_2[key].value
                else:
                    hyper_parameters_1[key].value = hyper_parameters_2[key].value
                    hyper_parameters_2[key].value = hyper_parameters_1[key].value
        parent_1.machine.set_hyper_parameters(hyper_parameters_1)
        child_1.set_machine(parent_1.machine)
        parent_2.machine.set_hyper_parameters(hyper_parameters_2)
        child_2.set_machine(parent_2.machine)
        return child_1, child_2

    def run(self):
        """Main function to run the genetic algorithm
        """
        self.create_initial_population(
            population_number=self._genetic_algorithm_parameters.number_population)
        for _ in range(self._genetic_algorithm_parameters.number_generation):
            self.selection()
            for i in range(self._genetic_algorithm_parameters.number_population // 2):
                child_1, child_2 = self.crossover(
                    parent_1=self._population[i],
                    parent_2=self._population[
                        self._genetic_algorithm_parameters.number_population - 1
                    ],
                )
                child_1.mutate_features_chromosome(
                    mutation_rate=self._genetic_algorithm_parameters.mutation_rate)
                child_2.mutate_features_chromosome(
                    mutation_rate=self._genetic_algorithm_parameters.mutation_rate)
                child_1.apply_dataset(file_machine=self._file_machine)
                child_2.apply_dataset(file_machine=self._file_machine)
                child_1.machine.mutate_hyper_parameters(
                    mutation_rate=self._genetic_algorithm_parameters.mutation_rate)
                child_2.machine.mutate_hyper_parameters(
                    mutation_rate=self._genetic_algorithm_parameters.mutation_rate)
                self._new_population.append(child_1)
                self._new_population.append(child_2)
            self.store_population()

    def export(self):
        """Export the best individual
        """
        csv_dict = []
        for individual in self._global_population:
            csv_dict.append(individual.to_dictionary())
        self._file_machine.storage_file.save_data_frame_to_csv(
            data_frame=csv_dict,
            prefix="optimization",
        )

    def selection(self):
        """Run every model in all population and sort by best metric
        """
        for individual in self._population:
            individual.run()
        self._population.sort(
            key=lambda individual: individual.index_metric, reverse=True)

    def store_population(self):
        """Store the population in a file
        """
        self._global_population += self._population
        self._global_population.sort(
            key=lambda individual: individual.index_metric, reverse=True)
        self._population = self._new_population
        self._new_population = []


def optimization_run(
    file_date: FileData,
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
        file_data=file_date,
        training_info=training_info,
        genetic_algorithm_parameters=genetic_algorithm_parameters)
    genetic_algorithm.run()
    genetic_algorithm.export()

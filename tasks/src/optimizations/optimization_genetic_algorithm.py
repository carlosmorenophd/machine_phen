"""Run a genetic algorithm for optimization
    """

import random
import copy

import pandas as pd

from src.optimizations.optimization_enum import (
    GeneticAlgorithmParameter,
)
from src.files.file_machine import (
    FileDataRegression,
    TrainingData,
    FileMachine,
    DatasetOptimizationData
)
from src.machines.machine_build import machine_build_regression_optimization_decimal
from src.machines.machine import MachineRegression
from src.metrics.metric import Metric
from src.metrics.metric_enums import MetricEnum
from src.machines.machine_enums import HyperTypeValueEnum


class GeneticIndividual():
    """Class to create a individual for genetic algorithm
    """

    def __init__(
        self,
        machine: MachineRegression = None,
        metric_selection: MetricEnum = None,
    ) -> None:
        self._dataset: DatasetOptimizationData = None
        self._machine = None
        if machine is not None:
            self._machine = machine
        self._features_chromosome:  list[bool] = None
        self._metric: Metric = None
        self._index_metric: float = None
        self._metric_selection = None
        if metric_selection is not None:
            self._metric_selection = metric_selection

    def __str__(self) -> str:
        return f"machine: {self._machine}  features: {self._dataset.features_name_that_be_true}"

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
        self._machine.training(
            x_train=self._dataset.x_train,
            y_train=self._dataset.y_train,
        )
        self._metric = self._machine.test(
            x_test=self._dataset.x_test,
            y_test=self._dataset.y_test,
        )
        if self._metric_selection is None:
            raise ValueError("Metric selection is not defined")
        self._index_metric = self._metric.get_single_metric(
            metric=self._metric_selection)

    def set_features_chromosome(self, features_chromosome: list[bool]):
        """Set the features chromosome"""
        self._features_chromosome = features_chromosome

    def set_machine(self, machine: MachineRegression):
        """Set the machine"""
        self._machine = machine

    def set_metric_selection(self, metric_selection: MetricEnum):
        """Set the metric selection"""
        self._metric_selection = metric_selection

    def apply_dataset(self, file_machine: FileMachine, features_chromosome: list[bool] = None):
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

    def to_dictionary(self, features_names: list[str]) -> dict:
        """Convert the individual to dictionary

        Returns:
            dict: Dictionary with the individual information
        """
        general_dict = {
            "machine_name": self._machine.machine_name,
            "index_metric": self._index_metric,
        }
        general_dict.update(self._metric.get_all_metric())
        for key in self._machine.hyper_parameters:
            new_key = f"{key}_{self._machine.machine_name}"
            general_dict[new_key] = self._machine.hyper_parameters[key].value
        for chromosome, feature in zip(self._features_chromosome, features_names):
            general_dict[f"feature_{feature}"] = chromosome
        return general_dict


class GeneticAlgorithm():
    """Main class to run the genetic algorithm for optimization process.
    """

    def __init__(
        self,
        file_data: FileDataRegression,
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
        self._message = ""

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
            machine=machine_build_regression_optimization_decimal(
                machine_name=random.choice(
                    self._genetic_algorithm_parameters.machines_key
                ),
                deep_decimal=self._genetic_algorithm_parameters.hyper_parameter_deep_decimal,
            ),
            metric_selection=self._genetic_algorithm_parameters.metric_selection,
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
        child_1 = GeneticIndividual(
            metric_selection=self._genetic_algorithm_parameters.metric_selection
        )
        child_2 = GeneticIndividual(
            metric_selection=self._genetic_algorithm_parameters.metric_selection
        )
        child_1.set_features_chromosome(parent_1.features_chromosome[
            :crossover_point] + parent_2.features_chromosome[crossover_point:])

        child_2.set_features_chromosome(parent_2.features_chromosome[
            :crossover_point] + parent_1.features_chromosome[crossover_point:])
        if parent_1.machine.machine_name == parent_2.machine.machine_name:
            hyper_parameters_1 = parent_1.machine.hyper_parameters
            hyper_parameters_2 = parent_2.machine.hyper_parameters
            for key, _ in parent_1.machine.hyper_parameters.items():
                if hyper_parameters_1[key].type_value == HyperTypeValueEnum.FLOAT:
                    value_1, value_2 = self.crossover_value_same_machine(
                        value_1=hyper_parameters_1[key].value,
                        value_2=hyper_parameters_2[key].value,
                        type_value=hyper_parameters_1[key].type_value,
                    )
                    hyper_parameters_1[key].set_value(value_1)
                    hyper_parameters_2[key].set_value(value_2)
                elif hyper_parameters_1[key].type_value == HyperTypeValueEnum.INT:
                    value_1, value_2 = self.crossover_value_same_machine(
                        value_1=hyper_parameters_1[key].value,
                        value_2=hyper_parameters_2[key].value,
                        type_value=hyper_parameters_1[key].type_value,
                    )
                    hyper_parameters_1[key].set_value(value_1)
                    hyper_parameters_2[key].set_value(value_2)
                elif hyper_parameters_1[key].type_value == HyperTypeValueEnum.CATEGORY:
                    if random.random() < self._genetic_algorithm_parameters.cross_over_rate:
                        hyper_parameters_1[key].set_value(
                            hyper_parameters_1[key].value)
                        hyper_parameters_2[key].set_value(
                            hyper_parameters_1[key].value)
                    else:
                        hyper_parameters_1[key].set_value(
                            hyper_parameters_2[key].value)
                        hyper_parameters_2[key].set_value(
                            hyper_parameters_1[key].value)
            parent_1.machine.set_hyper_parameters(hyper_parameters_1)
            child_1.set_machine(parent_1.machine)
            parent_2.machine.set_hyper_parameters(hyper_parameters_2)
            child_2.set_machine(parent_2.machine)
        else:
            child_1.set_machine(parent_1.machine)
            child_2.set_machine(parent_2.machine)
            child_2.machine.force_mutate_hyper_parameters(
                deep_decimal=self._genetic_algorithm_parameters.hyper_parameter_deep_decimal
            )
            if random.random() < self._genetic_algorithm_parameters.cross_over_rate:
                child_1.machine.force_mutate_hyper_parameters(
                    deep_decimal=self._genetic_algorithm_parameters.hyper_parameter_deep_decimal
                )
        return child_1, child_2

    def crossover_value_same_machine(
            self,
            value_1: str,
            value_2:  str,
            type_value: HyperTypeValueEnum
    ) -> tuple[str, str]:
        """Cross over between two values of hyper parameters

        Args:
            value_1 (str): value of hyper parameter 1
            value_2 (str): value of hyper parameter 2
            type_value (HyperTypeValueEnum): type of hyper parameter

        Returns:
            tuple[str, str]: tuple of two new values of hyper parameters
        """
        mean_value = self.calculate_mean_str_hyper_parameter(
            value_1=value_1, value_2=value_2, type_value=type_value)
        if random.random() < self._genetic_algorithm_parameters.cross_over_rate:
            return mean_value, mean_value
        return value_1, mean_value

    def calculate_mean_str_hyper_parameter(
            self,
            value_1: str,
            value_2:  str,
            type_value: HyperTypeValueEnum
    ) -> str:
        """Calculate the mean between two str values

        Args:
            hyper_parameter_1 (str): First value of hyper parameter
            hyper_parameter_2 (str): Second value of hyper parameter

        Returns:
            str: Mean value between two values of hyper parameters
        """
        if type_value == HyperTypeValueEnum.INT:
            return str(
                int(
                    int(value_1) +
                    int(value_2) / 2
                )
            )
        if type_value == HyperTypeValueEnum.FLOAT:
            return str(round(
                float(value_1) +
                float(value_2) / 2,
                self._genetic_algorithm_parameters.hyper_parameter_deep_decimal
            ))
        raise ValueError("Type value not defined")

    def run(self):
        """Main function to run the genetic algorithm
        """
        self.create_initial_population(
            population_number=self._genetic_algorithm_parameters.number_population
        )
        self.initial_log()
        for generation_number in range(self._genetic_algorithm_parameters.number_generation):
            self.progress_log(generation_number=generation_number)
            self.selection()
            for i in range(self._genetic_algorithm_parameters.number_population // 2):
                child_1, child_2 = self.crossover(
                    parent_1=self._population[i],
                    parent_2=self._population[
                        self._genetic_algorithm_parameters.number_population - 1
                    ],
                )
                child_1, child_2 = self.mutation_two_children(
                    child_1=child_1,
                    child_2=child_2,
                )
                self._new_population.append(child_1)
                self._new_population.append(child_2)
            self.store_population()
            self.export()

    def initial_log(self):
        """Initial log of algorithm
        """
        self.adding_message(message="Starting the genetic algorithm")
        self.adding_message(
            message=f"population: {self._genetic_algorithm_parameters.number_population}")
        self.adding_message(
            message=f"generations: {self._genetic_algorithm_parameters.number_generation}")
        self.log_message()

    def progress_log(self, generation_number: int):
        """Log of the generation
        """
        self.adding_message(message=f"Progress: {
            generation_number / self._genetic_algorithm_parameters.number_generation
        }")
        self.log_message()

    def mutation_two_children(self, child_1: GeneticIndividual, child_2: GeneticIndividual):
        "Mutation of the population"
        child_1.mutate_features_chromosome(
            mutation_rate=self._genetic_algorithm_parameters.mutation_rate)
        child_2.mutate_features_chromosome(
            mutation_rate=self._genetic_algorithm_parameters.mutation_rate)
        child_1.apply_dataset(file_machine=self._file_machine)
        child_2.apply_dataset(file_machine=self._file_machine)
        child_1.machine.mutate_hyper_parameters(
            mutation_rate=self._genetic_algorithm_parameters.mutation_rate,
            deep_decimal=self._genetic_algorithm_parameters.hyper_parameter_deep_decimal,
        )
        child_2.machine.mutate_hyper_parameters(
            mutation_rate=self._genetic_algorithm_parameters.mutation_rate,
            deep_decimal=self._genetic_algorithm_parameters.hyper_parameter_deep_decimal,
        )
        child_1 = self.mutate_machine(child=child_1)
        child_2 = self.mutate_machine(child=child_2)
        return child_1, child_2

    def mutate_machine(self, child: GeneticIndividual):
        """Mutate machine

        Args:
            child_1 (GeneticIndividual): Child to mutate machine
        """
        if random.random() < self._genetic_algorithm_parameters.mutate_machine:
            child.set_machine(machine=machine_build_regression_optimization_decimal(
                machine_name=random.choice(
                    self._genetic_algorithm_parameters.machines_key
                ),
                deep_decimal=self._genetic_algorithm_parameters.hyper_parameter_deep_decimal,
            ),
            )
        return child

    def export(self):
        """Export the best individual
        """
        metric_to_csv = []
        for individual in self._global_population:
            metric_dict = individual.to_dictionary(
                features_names=self._file_machine.columns_name_with_out_target,
            )
            metric_to_csv.append(metric_dict)
        data_frame_metric = pd.DataFrame(metric_to_csv)
        self._file_machine.storage_file.save_data_frame_to_csv(
            data_frame=data_frame_metric,
            prefix="optimization",
        )
        ten_percent = int(len(data_frame_metric) * 0.1)
        sampled_data_frame = data_frame_metric.sample(n=ten_percent)
        self._file_machine.storage_file.save_data_frame_to_csv(
            data_frame=sampled_data_frame,
            prefix="optimization_overlap",
        )

    def selection(self):
        """Run every model in all population and sort by best metric
        """
        for number_individual, individual in enumerate(self._population):
            # self.selection_log(number_individual=number_individual)
            individual.run()
        self._population.sort(
            key=lambda individual: individual.index_metric,
            reverse=self.selection_reverse()
        )

    def selection_reverse(self) -> bool:
        """Get if the best model is the lowest or the highest

        Raises:
            ValueError: Model no define

        Returns:
            _type_: sort ascending or descending
        """
        if self._genetic_algorithm_parameters.metric_selection in [
            MetricEnum.ACCURACY_MEAN_ABSOLUTE_PERCENTAGE_ERROR,
            MetricEnum.R2_SCORE,
        ]:
            return False
        if self._genetic_algorithm_parameters.metric_selection in [
            MetricEnum.MEAN_ABSOLUTE_ERROR,
            MetricEnum.MEAN_SQUARED_ERROR
        ]:
            return True
        raise ValueError("Metric not defined, to select the best model")

    def selection_log(self, number_individual: int):
        """Log the selection
        """
        self.adding_message(
            message=f"Individual {number_individual} ")
        self.adding_message(
            message=f" - {len(self._population)} ")
        self.log_message()

    def store_population(self):
        """Store the population in a file
        """
        self._global_population += copy.deepcopy(self._population)
        self._population = []
        self._global_population.sort(
            key=lambda individual: individual.index_metric, reverse=True)
        self._population = copy.deepcopy(self._new_population)
        self._new_population = []

    def adding_message(self, message: str, prefix: bool = False) -> None:
        """Adding message to the optimization
        """
        if prefix is False:
            self._message = f"{self._message} {message}"
        else:
            self._message = f"{message} {self._message}"

    def log_message(self, quick_message: str = None) -> None:
        """Log the message
        """
        if quick_message is not None:
            print(quick_message)
        else:
            print(self._message)
            self._message = ""

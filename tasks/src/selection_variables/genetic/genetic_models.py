"""Genetic algorithm to run it"""
import random
import numpy as np
import pandas as pd

from src.machines.machine_enums import MachineJson, FileAccessRunnerProperties
from src.machines.machine_data_frame_handler import DataFrameHandler
from src.machines.regression_run import machine_build_regression
from src.metrics.metric_enums import MetricEnum
from src.selection_variables.selection_data_frame_handler import SelectionBestMetric
from src.selection_variables.force_brute import combination_columns_from_data_frame
from src.selection_variables.genetic.genetic_enum import GeneticParameter, MachineMainRegression


class GeneticAlgorithm():
    """Model simple genetic algorithm
    """

    def __init__(self, machine_main: MachineMainRegression,
                 parameter: GeneticParameter = GeneticParameter()) -> None:
        self.machine_main = machine_main
        self.parameter = parameter
        self.create_initial_population()

    def create_initial_population(self):
        """Create the population initial

        Args:
            parameters (GeneticParameter): Parameters of genetic algorithms

        Returns:
            list: initial population
        """
        self.population = []
        for _ in range(self.parameter.population_size):
            individual = [random.randint(0, 1)
                          for _ in range(self.parameter.chromosome_length)]
            self.population.append(individual)


def genetic_algorithm(
        machine_main: MachineMainRegression,
        parameter: GeneticParameter = GeneticParameter(),
) -> SelectionBestMetric:
    """Launched genetic algorithm

    Args:
        df (pd.DataFrame): dataset to work with it
        selection (SelectionBestMetric): class to store result of individuals
        machine (MachineJson): machine to use
        parameter (GeneticParameter, optional): parameter for genetic algorithms. 
            Defaults to GeneticParameter().

    Returns:
        SelectionBestMetric: return all metric of algorithms
    """

    for generation in range(parameter.num_generations):
        print(f"Generation => {generation} of {parameter.num_generations}")
        selected_population = selection(
            population=population, machine_main=machine_main)
        new_population = []
        for i in range(0, parameter.population_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i+1]
            child1, child2 = crossover(
                parent1, parent2, parameter.crossover_rate)
            child1 = mutation(child1, parameter.mutation_rate)
            child2 = mutation(child2, parameter.mutation_rate)
            new_population.extend([child1, child2])

        df_metric = machine_main.best_metric.metric_values
        machine_main.data_frame.storage_file.save_data_frame_to_csv(
            data_frame=df_metric, prefix="metric_genetic_selection",
        )
        population = new_population
    return machine_main.best_metric


def fitness_function(
        individual: list,
        machine_main: MachineMainRegression,
) -> float:
    """A simple fitness function, you can replace this with your specific function

    Args:
        individual (list): chromosome individual
        machine_main (MachineMainRegression): machine main definition

    Returns:
        float: _description_
    """
    combination_bool = [bool(x) for x in individual]
    combination = machine_main.data_frame.get_data_frame_without_target(
    ).columns[combination_bool]
    machine_main.data_frame.change_column_from_data_frame(
        columns_to_keep=combination)
    dataset = machine_main.data_frame.dataset
    machine_main.machine.build_machine()
    machine_main.machine.training(dataset.x_train, dataset.y_train)
    error_metric = machine_main.machine.test(
        x_test=dataset.x_test,
        y_test=dataset.y_test,
    )
    fitness = machine_main.best_metric.add_metric_value(
        machine_definition=machine_main.machine_definition,
        columns=combination,
        metrics=error_metric.metrics,
    )

    return fitness


def selection(population, machine_main: MachineMainRegression):
    """Selection the best population"""
    selected_individuals = []
    tournament_size = 2
    for _ in range(len(population)):
        tournament_participants = random.sample(population, tournament_size)
        selected_individuals.append(
            max(tournament_participants, key=lambda x: fitness_function(
                individual=x,
                machine_main=machine_main,
            )))
    return selected_individuals


def crossover(parent1, parent2, crossover_rate):
    """Generate new individual

    Args:
        parent1 (_type_): _description_
        parent2 (_type_): _description_
        crossover_rate (_type_): _description_

    Returns:
        _type_: _description_
    """
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    return parent1, parent2


def mutation(individual: list, mutation_rate: float) -> list:
    """Mutation new individual

    Args:
        individual (list): individual of population 
        mutation_rate (float): range of mutate

    Returns:
        list: individual mutated
    """
    for index, _ in enumerate(individual):
        if random.random() < mutation_rate:
            individual[index] = 1 - individual[index]
    return individual

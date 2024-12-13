"""Runner to get variable"""
import random


import numpy as np
import pandas as pd

from src.machines.machine_enums import MachineJson, FileAccessRunnerProperties
from src.machines.machine_data_frame_handler import DataFrameHandler
from src.machines.regression_run import machine_build_regression, MachineRegression
from src.metrics.metric_enums import MetricEnum
from src.selection_variables.selection_data_frame_handler import SelectionBestMetric
from src.selection_variables.force_brute import combination_columns_from_data_frame
from src.selection_variables.genetic.genetic_enum import GeneticParameter


def selection_force_brute_run(
    file_access_runner: FileAccessRunnerProperties,
    machine_definition: MachineJson,
    metric: MetricEnum = MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
) -> None:
    """Run forward selection with force brute
    Args:
            initial_column (_type_): _description_
            metric (MetricEnum, optional): metric to validate machine.
                Defaults to MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR.
    """
    data_frame = DataFrameHandler(file_access_runner=file_access_runner)
    best_metric = SelectionBestMetric(metric_to_evaluate=metric)
    df = data_frame.get_data_frame_without_target()
    machine = machine_build_regression(machine_definition=machine_definition)
    for combination in combination_columns_from_data_frame(df):
        data_frame.change_column_from_data_frame(columns_to_keep=combination)
        dataset = data_frame.dataset
        machine.build_machine()
        machine.training(dataset.x_train, dataset.y_train)
        error_metric = machine.test(
            x_test=dataset.x_test,
            y_test=dataset.y_test,
        )
        best_metric.add_metric_value(
            machine_definition=machine_definition,
            columns=combination,
            metrics=error_metric.metrics,
        )
    df_metric = best_metric.metric_values
    data_frame.storage_file.save_data_frame_to_csv(
        data_frame=df_metric, prefix="metric_forward_selection",
    )


def selection_evolution_strategy_run(
        file_access_runner: FileAccessRunnerProperties,
    machine_definition: MachineJson,
    metric: MetricEnum = MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
) -> None:
    """Search best combination of variables on dataset and 

    Args:
        file_access_runner (FileAccessRunnerProperties): file access
        machine_definition (MachineJson): machine
        metric (MetricEnum, optional): metric to evaluate. 
            Defaults to MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR.
    """
    data_frame = DataFrameHandler(file_access_runner=file_access_runner)
    best_metric = SelectionBestMetric(metric_to_evaluate=metric)

    best_solution = genetic_algorithm(
        data_frame=data_frame,
        best_metric=best_metric,
        machine_definition=machine_definition,
    )
    print(best_solution)


def genetic_algorithm(
        data_frame: DataFrameHandler,
        best_metric:  SelectionBestMetric,
        machine_definition: MachineJson,
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

    parameter.chromosome_length = len(
        data_frame.get_data_frame_without_target.columns)
    machine = machine_build_regression(machine_definition=machine_definition)

    population = create_initial_population(
        parameter
    )

    for generation in range(parameter.num_generations):
        print(f"Generation => {generation}")
        fitness_scores = [fitness_function(
            individual=individual,
            best_metric=best_metric,
            machine=machine,
            machine_definition=machine_definition,
            data_frame=data_frame,
        ) for individual in population]
        selected_population = selection(population, fitness_scores)

        new_population = []
        for i in range(0, parameter.population_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i+1]
            child1, child2 = crossover(
                parent1, parent2, parameter.crossover_rate)
            child1 = mutation(child1, parameter.mutation_rate)
            child2 = mutation(child2, parameter.mutation_rate)
            new_population.extend([child1, child2])

        population = new_population
    return best_metric


def fitness_function(
        individual,
        machine_definition: MachineJson,
        best_metric: SelectionBestMetric,
        data_frame: DataFrameHandler,
        machine: MachineRegression,
) -> float:
    """A simple fitness function, you can replace this with your specific function"""
    combination = data_frame.get_data_frame_without_target[individual]
    data_frame.change_column_from_data_frame(columns_to_keep=combination)
    dataset = data_frame.dataset
    machine.build_machine()
    machine.training(dataset.x_train, dataset.y_train)
    error_metric = machine.test(
        x_test=dataset.x_test,
        y_test=dataset.y_test,
    )
    fitness = best_metric.add_metric_value(
        machine_definition=machine_definition,
        columns=combination,
        metrics=error_metric.metrics,
    )

    return fitness


def create_initial_population(parameters: GeneticParameter) -> list:
    """Create the population initial

    Args:
        parameters (GeneticParameter): Parameters of genetic algorithms

    Returns:
        list: initial population
    """
    population = []
    for _ in range(parameters.population_size):
        individual = [random.randint(0, 1)
                      for _ in range(parameters.chromosome_length)]
        population.append(individual)
    return population


def selection(population, fitness_scores):
    """Selection the best population"""
    selected_individuals = []
    for _ in range(len(population)):
        tournament_size = 2
        tournament_participants = random.sample(population, tournament_size)
        selected_individuals.append(
            max(tournament_participants, key=fitness_scores))
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
    else:
        return parent1, parent2


def mutation(individual, mutation_rate):
    """Mutation new individual

    Args:
        individual (_type_): _description_
        mutation_rate (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

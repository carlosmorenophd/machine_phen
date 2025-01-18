"""Genetic algorithm to run it"""
import random
from dataclasses import dataclass

from src.selection_variables.selection_data_frame_handler import SelectionBestMetric
from src.selection_variables.genetic.genetic_enum import GeneticParameter, MachineMainRegression

@dataclass
class GeneticIndividual():
    """Individual for genetic
    """
    features: list[bool]
    machine: MachineMainRegression


class GeneticAlgorithm():
    """Model simple genetic algorithm
    """
    def __init__(
        self,
        machine_main: MachineMainRegression,
        parameter: GeneticParameter = GeneticParameter()
    ) -> None:
        self.machine_main = machine_main
        self.parameter = parameter
        self.population = []
        self.create_initial_population()

    def create_initial_population(self):
        """Create the population initial

        Args:
            parameters (GeneticParameter): Parameters of genetic algorithms

        Returns:
            list: initial population
        """
        for _ in range(self.parameter.population_size):
            individual = [random.randint(0, 1)
                          for _ in range(self.parameter.chromosome_length)]
            self.population.append(individual)

    def run(
            self,
    ) -> SelectionBestMetric:
        """Launched genetic algorithm
        Returns:
            SelectionBestMetric: return all metric of algorithms
        """

        for generation in range(self.parameter.num_generations):
            print(f"Generation => {generation} of {self.parameter.num_generations}")
            selected_population = self.selection()
            new_population = []
            for i in range(0, self.parameter.population_size, 2):
                parent1, parent2 = selected_population[i], selected_population[i+1]
                child1, child2 = self.crossover(
                    parent1,
                    parent2,
                )
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_population.extend([child1, child2])

            df_metric = self.machine_main.best_metric.metric_values
            self.machine_main.data_frame.storage_file.save_data_frame_to_csv(
                data_frame=df_metric, prefix="metric_genetic_selection",
            )
            self.population = new_population
        return self.machine_main.best_metric

    def fitness_function(
            self,
            individual: list,
    ) -> float:
        """A simple fitness function, you can replace this with your specific function

        Args:
            individual (list): chromosome individual
            machine_main (MachineMainRegression): machine main definition

        Returns:
            float: _description_
        """
        combination_bool = [bool(x) for x in individual]
        combination = self.machine_main.data_frame.get_data_frame_without_target(
        ).columns[combination_bool]
        exist = self.machine_main.best_metric.exist_this_columns(
            columns=combination,
            machine_definition=self.machine_main.machine_definition,
        )
        if exist is not None:
            return exist
        self.machine_main.data_frame.change_column_from_data_frame(
            columns_to_keep=combination)
        dataset = self.machine_main.data_frame.dataset
        self.machine_main.machine.build_machine()
        self.machine_main.machine.training(dataset.x_train, dataset.y_train)
        error_metric = self.machine_main.machine.test(
            x_test=dataset.x_test,
            y_test=dataset.y_test,
        )
        fitness = self.machine_main.best_metric.add_metric_value(
            machine_definition=self.machine_main.machine_definition,
            columns=combination,
            metrics=error_metric.metrics,
        )

        return fitness

    def selection(self):
        """Selection the best population"""
        selected_individuals = []
        tournament_size = 2
        for _ in range(len(self.population)):
            tournament_participants = random.sample(
                self.population, tournament_size)
            selected_individuals.append(
                max(tournament_participants, key=lambda x: self.fitness_function(
                    individual=x,
                )))
        return selected_individuals

    def crossover(self, parent1, parent2):
        """Generate new individual

        Args:
            parent1 (_type_): _description_
            parent2 (_type_): _description_
            crossover_rate (_type_): _description_

        Returns:
            _type_: _description_
        """
        if random.random() < self.parameter.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        return parent1, parent2

    def mutation(
            self,
            individual: list,
    ) -> list:
        """Mutation new individual

        Args:
            individual (list): individual of population 
            mutation_rate (float): range of mutate

        Returns:
            list: individual mutated
        """
        for index, _ in enumerate(individual):
            if random.random() < self.parameter.mutation_rate:
                individual[index] = 1 - individual[index]
        return individual

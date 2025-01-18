"""Operative class to run the genetic algorithm.
"""

from src.machines.machine_data_frame_handler import FileAccessForMachine
from src.optimizations.optimization_enum import GeneticAlgorithmParameter

class GeneticAlgorithm():
    """Main class to run the genetic algorithm for optimization process.
    """

    def __init__(self, file_machine: FileAccessForMachine, genetic_algorithm_parameters:  GeneticAlgorithmParameter ) -> None:
        self._file_machine = file_machine
        self._genetic_algorithm_parameters = genetic_algorithm_parameters

    def run(self):
        """Run the genetic algorithm
        """
         
    
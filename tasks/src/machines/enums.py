"""Skelton for class to predict"""
from dataclasses import dataclass

from enum import Enum


class SupportVectorRegressionKernelEnum(Enum):
    """Kernel for SVR

    Args:
        Enum (_type_): _description_
    """
    RBF = 'rbf'
    LINEAR = 'linear'
    POLY = 'poly'
    SIGMOID = 'sigmoid'
    PRE_COMPUTER = 'precomputed'

MachineNames = Enum('MachineNames', [("RF", "random_forest")])


@dataclass
class MachineJson:
    """Generic class for Machines Json
    """
    name: MachineNames


@dataclass
class RandomForestJson(MachineJson):
    """Generic class for Machines Json
    """
    n_estimators: int = 1000
    random_state: int = 42
    n_jobs: int = -1

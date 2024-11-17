"""Skelton for class to predict"""
from dataclasses import dataclass
from typing import Dict


from enum import Enum
import pandas as pd
from src.metrics.metric_enums import MetricEnum


class SupportVectorRegressionKernelEnum(Enum):
    """Kernel for SVR

    Args:
        Enum (_type_): _description_
    """
    LINEAR = 'linear'
    POLY = 'poly'
    PRE_COMPUTER = 'precomputed'
    RBF = 'rbf'
    SIGMOID = 'sigmoid'


def cast_kernel_svr(input_kernel: str):
    """Function to cast text to support vector regression kernel enum

    Args:
        input (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if input_kernel == SupportVectorRegressionKernelEnum.LINEAR.value:
        return SupportVectorRegressionKernelEnum.LINEAR
    if input_kernel == SupportVectorRegressionKernelEnum.POLY.value:
        return SupportVectorRegressionKernelEnum.POLY
    if input_kernel == SupportVectorRegressionKernelEnum.PRE_COMPUTER.value:
        return SupportVectorRegressionKernelEnum.PRE_COMPUTER
    if input_kernel == SupportVectorRegressionKernelEnum.RBF.value:
        return SupportVectorRegressionKernelEnum.RBF
    if input_kernel == SupportVectorRegressionKernelEnum.SIGMOID.value:
        return SupportVectorRegressionKernelEnum.SIGMOID
    raise ValueError(f"Not kernel valid {input_kernel}")


MachineNames = Enum(
    'MachineNames', [
        ("RF", "random_forest"),
        ("XGB", "extreme_gradient_boost"),
        ("BAP", "bayesian_prediction"),
        ("LAP", "lasso_prediction"),
        ("SVRP", "support_vector_regression_prediction"),
    ]
)


@dataclass
class MachineJson:
    """Generic class for Machines Json
    """
    name: MachineNames

    def __str__(self) -> str:
        return f"machine_name: {
            self.name.value
        }"


@dataclass
class RandomForestJson(MachineJson):
    """Basic parameters for Random Forest machine definition
    """
    name: MachineNames = MachineNames.RF
    n_estimators: int = 1000
    random_state: int = 42
    n_jobs: int = -1

    def __str__(self) -> str:
        return f"machine_name: {
            self.name.value
        } - [ n_estimators: {
            self.n_estimators
        }, random_state: {
            self.random_state
        }, n_jobs: {
            self.n_jobs
        } ]"


@dataclass
class ExtremeGradientBoost(MachineJson):
    """Basic parameters for Random Forest machine definition
    """
    name: MachineNames = MachineNames.XGB
    n_estimators: int = -1
    max_depth: int = 0
    max_leaves: int = 0

    def __str__(self) -> str:
        return f"machine_name: {
            self.name.value
        } - [ n_estimators: {
            self.n_estimators
        }, max_depth: {
            self.max_depth
        }, max_leaves: {
            self.max_leaves
        } ]"


@dataclass
class BayesianPredictionDefinition(MachineJson):
    """Basic parameters for Random Forest machine definition
    """
    name: MachineNames = MachineNames.BAP

    def __str__(self) -> str:
        return f"machine_name: {
            self.name.value
        } - [ ]"


@dataclass
class LassoPredictionDataClass(MachineJson):
    """Basic parameters for Random Forest machine definition
    """
    name: MachineNames = MachineNames.LAP
    alpha: float = 0.1

    def __str__(self) -> str:
        return f"machine_name: {
            self.name.value
        } - [ ]"


@dataclass
class SupportVectorRegressionPredictionDataClass(MachineJson):
    """Basic parameters for Random Forest machine definition
    """
    name: MachineNames = MachineNames.SVRP
    kernel: SupportVectorRegressionKernelEnum = SupportVectorRegressionKernelEnum.LINEAR
    c: float = 1.0
    epsilon: float = 0.1

    def __str__(self) -> str:
        return f"machine_name: {
            self.name.value
        } - [kernel: {
            self.kernel.value
        }, c: {
            self.c
        }, epsilon: {
            self.epsilon
        } ]"



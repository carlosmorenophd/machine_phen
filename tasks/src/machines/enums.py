"""Skelton for class to predict"""
from dataclasses import dataclass

from enum import Enum


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


@dataclass
class RandomForestJson(MachineJson):
    """Basic parameters for Random Forest machine definition
    """
    name: MachineNames = MachineNames.RF
    n_estimators: int = 1000
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class ExtremeGradientBoost(MachineJson):
    """Basic parameters for Random Forest machine definition
    """
    name: MachineNames = MachineNames.XGB
    n_estimators: int = -1
    max_depth: int = 0
    max_leaves: int = 0


@dataclass
class BayesianPredictionDefinition(MachineJson):
    """Basic parameters for Random Forest machine definition
    """
    name: MachineNames = MachineNames.BAP


@dataclass
class LassoPredictionDataClass(MachineJson):
    """Basic parameters for Random Forest machine definition
    """
    name: MachineNames = MachineNames.LAP
    alpha: float = 0.1


@dataclass
class SupportVectorRegressionPredictionDataClass(MachineJson):
    """Basic parameters for Random Forest machine definition
    """
    name: MachineNames = MachineNames.SVRP
    kernel: SupportVectorRegressionKernelEnum = SupportVectorRegressionKernelEnum.LINEAR
    c: float = 1.0
    epsilon: float = 0.1

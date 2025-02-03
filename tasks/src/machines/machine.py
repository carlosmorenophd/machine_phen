"""To run machine to prediction"""

import random
import time
import json
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta

from numpy import ndarray
# from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
# from sklearn.svm import SVR
from xgboost import XGBRegressor

# from src.machines.machine_enums import (SupportVectorKernelEnum)
from src.metrics.metric import Metric
from src.machines.machine_enums import MachineNames

HyperTypeValueEnum = Enum('HyperTypeValueEnum', [
    ('FLOAT', 'float'),
    ('INT', 'int'),
    ('CLASS', 'class'),
])


@dataclass
class HyperParametersDefinition:
    """General definition for hyperparameters
    """
    value = None
    low: float = None
    high: float = None
    type_value: HyperTypeValueEnum = HyperTypeValueEnum.FLOAT


class MachineRegression(ABC):
    """Machine for regression and prediction
    """

    def __init__(self) -> None:
        self._time_training = 0
        self._machine = None
        self._machine_name: MachineNames = None
        self._error_metric = None
        self._hyper_parameters: dict[HyperParametersDefinition] = None
        self._default_hyper_parameters: dict[HyperParametersDefinition] = None

    @property
    def hyper_parameters(self) -> list[HyperParametersDefinition]:
        """Create a random hyper parameters
        """
        return self._hyper_parameters

    @property
    def machine_name(self) -> MachineNames:
        """Get the machine name by enum"""
        return self._machine_name

    @abstractmethod
    def build_machine(self) -> None:
        """Create a machine for training and predict
        """

    def identity(self) -> dict:
        """Return the str that identify the machine and the features
        """
        identity_dict = {
            "name": self._machine_name,
        }
        identity_dict.update(self._hyper_parameters)
        return identity_dict

    def __str__(self):
        """Create str with name and hyper parameters"""
        return json.dumps(self.identity())

    def set_hyper_parameters(self, hyper_parameters: list[HyperParametersDefinition]) -> None:
        """Set hyper parameters to machine
        """
        self._hyper_parameters = hyper_parameters

    def training(self, x_train: ndarray, y_train: ndarray) -> None:
        """Training function

        Args:
            x_train (ndarray): Vector for training
            y_train (ndarray): Vector to predict
        """
        start = time.time()
        self._machine.fit(x_train, y_train)
        self._time_training = timedelta(seconds=time.time() - start)

    def test(
        self,
        x_test: ndarray,
        y_test: ndarray,
    ) -> Metric:
        """Run test on machine and store data on error

        Args:
            x_test (ndarray): _description_
            y_test (ndarray): _description_
        """
        return Metric(
            x_test=x_test,
            y_test=y_test,
            y_predicted=self._machine.predict(x_test)
        )

    def prediction(self, x_test: ndarray) -> ndarray:
        """Predict new values

        Args:
            x_test (ndarray): Vector for predict

        Returns:
            ndarray: Vector predicted
        """
        return self._machine.predict(x_test)

    def force_mutate_hyper_parameters(self, deep_decimal: int) -> None:
        """Force mutate the hyper parameters

        Args:
            deep_decimal (int): Decimal to round the hyper parameters
        """
        for key, _ in self._hyper_parameters.items():
            self._hyper_parameters[key].value = self.generate_random_hyperparameter(
                low=self._hyper_parameters[key].low,
                high=self._hyper_parameters[key].high,
                deep_decimal=deep_decimal,
                type_value=self._hyper_parameters[key].type_value,
            )

    def mutate_hyper_parameters(self, mutation_rate: float, deep_decimal: int):
        """Mutate the hyper parameters

        Args:
            mutation_rate (float): Range between 0 and 1 to mutate the hyper parameters
        """
        for key, _ in self._hyper_parameters.items():
            if random.random() < mutation_rate:
                self._hyper_parameters[key].value = self.generate_random_hyperparameter(
                    low=self._hyper_parameters[key].low,
                    high=self._hyper_parameters[key].high,
                    deep_decimal=deep_decimal,
                    type_value=self._hyper_parameters[key].type_value,
                )

    def generate_random_hyperparameter(
            self,
            low: float,
            high: float,
            deep_decimal: int,
            type_value: HyperTypeValueEnum,
    ) -> float:
        """Generate a random hyperparameter value between low and high

        Args:
            low (float): Lower bound of the range
            high (float): Upper bound of the range

        Returns:
            float: Randomly generated hyperparameter value
        """
        if type_value == HyperTypeValueEnum.FLOAT:
            return round(random.uniform(low, high), deep_decimal)
        if type_value == HyperTypeValueEnum.INT:
            return int(round(random.uniform(low, high), deep_decimal))
        raise ValueError("Not valid type of value")


class BayesianRegression(MachineRegression):
    """Class to run a Bayesian Prediction
    """

    def __init__(self, is_default_parameters: bool = True) -> None:
        super().__init__()
        self._machine_name = MachineNames.BAYESIAN_REGRESSION.value
        self._default_hyper_parameters = {}
        self._hyper_parameters = {}
        self._default_hyper_parameters["alpha_1"] = HyperParametersDefinition(
            low=1e-6,
            high=1e-1,
            value=0.000001,
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._default_hyper_parameters["lambda_1"] = HyperParametersDefinition(
            low=1e-6,
            high=1e-1,
            value=0.000001,
            type_value=HyperTypeValueEnum.FLOAT,
        )
        if is_default_parameters:
            self._hyper_parameters = self._default_hyper_parameters

    def build_machine(self) -> None:
        self._machine = BayesianRidge(
            alpha_1=self._hyper_parameters["alpha_1"].value,
            lambda_1=self._hyper_parameters["lambda_1"].value
        )


# class LinearRRegression(MachineRegression):
#     """Machine for linear regression
#     """

#     def build_hyper_parameters_random(self) -> None:
#         self._machine = LinearRegression()

#     def __str__(self) -> str:
#         return f"Linear regression - {1} "


# class RidgeRegression(MachineRegression):
#     """Machine for ridge regression

#     Args:
#         MachinePrediction (_type_): Abstract method
#     """

#     def __init__(self, alpha: float = 0.1) -> None:
#         super().__init__()
#         self.alpha = alpha

#     def build_hyper_parameters_random(self) -> None:
#         self._machine = Ridge(alpha=self.alpha)

#     def build_hyper_parameters_random(self) -> list:
#         return [
#             {
#                 "alpha": self.generate_random_hyperparameter(),
#             }
#         ]

#     def __str__(self) -> str:
#         return f"Ridge - {self.alpha} "


# class LassoRegression(MachineRegression):
#     """Machine for linear LASSO

#     Args:
#         MachinePrediction (_type_): Abstract method
#     """

#     def __init__(self, alpha: float = 0.1) -> None:
#         super().__init__()
#         self.alpha = alpha

#     def build_hyper_parameters_random(self) -> None:
#         self._machine = Lasso(alpha=self.alpha)

#     def build_hyper_parameters_random(self) -> list:
#         return [
#             {
#                 "alpha": self.generate_random_hyperparameter(),
#             }
#         ]

#     def __str__(self) -> str:
#         return f"LASSO - {self.alpha} "


# class RandomForestRegression(MachineRegression):
#     """Machine for Random forest

#     Args:
#         MachinePrediction (_type_): Abstract method
#     """

#     def __init__(self, n_estimators: int = 1000, random_sate: int = 42, n_jobs: int = -1) -> None:
#         super().__init__()
#         self.n_estimators = n_estimators
#         self.random_sate = random_sate
#         self.n_jobs = n_jobs

#     def build_hyper_parameters_random(self) -> None:
#         self.machine = RandomForestRegressor(
#             n_estimators=self.n_estimators,
#             random_state=self.random_sate,
#             n_jobs=self.n_jobs,
#         )

#     def __str__(self) -> str:
#         return f"Random Forest Prediction - {self.n_estimators} "


# class SupportVectorRegression(MachineRegression):
#     """Machine for Support Vector Regression

#     Args:
#         MachinePrediction (_type_): Abstract method
#     """

#     def build_hyper_parameters_random(self) -> None:
#         self.machine = SVR(
#             kernel=self.kernel.value,
#             C=self.c,
#             epsilon=self.epsilon,
#         )

#     def build_hyper_parameters_random(self) -> list:
#         return [
#             {
#                 "C": self.generate_random_hyperparameter(),
#                 "epsilon": self.generate_random_hyperparameter(),
#             }
#         ]

#     def __str__(self) -> str:
#         return f"Support Vector Regression - {self.kernel.value}"

#     def __init__(
#         self,
#         kernel: SupportVectorKernelEnum = SupportVectorKernelEnum.LINEAR,
#         c: float = 1.0,
#         epsilon: float = 0.1,
#     ) -> None:
#         super().__init__()
#         self.kernel = kernel
#         self.c = c
#         self.epsilon = epsilon


class ExtremeGradientBoostRegression(MachineRegression):
    """Machine for prediction on Extreme Gradient Boosting

    Args:
        MachinePrediction (_type_): Abstract method
    """

    def __init__(self) -> None:
        super().__init__()
        self._machine_name = MachineNames.EXTREME_GRADIENT_BOOSTING_REGRESSION.value
        self._default_hyper_parameters = {}
        self._hyper_parameters = {}
        self._default_hyper_parameters["n_estimators"] = HyperParametersDefinition(
            low=10,
            high=10000,
            value=100,
            type_value=HyperTypeValueEnum.INT,
        )
        self._default_hyper_parameters["eta"] = HyperParametersDefinition(
            low=0,
            high=1,
            value=0.3,
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._default_hyper_parameters["gamma"] = HyperParametersDefinition(
            low=0,
            high=1,
            value=0.3,
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._default_hyper_parameters["max_depth"] = HyperParametersDefinition(
            low=0,
            high=300,
            value=6,
            type_value=HyperTypeValueEnum.INT,
        )
        self._default_hyper_parameters["min_child_weight"] = HyperParametersDefinition(
            low=0,
            high=300,
            value=1,
            type_value=HyperTypeValueEnum.INT,
        )
        self._default_hyper_parameters["max_delta_step"] = HyperParametersDefinition(
            low=0,
            high=300,
            value=0,
            type_value=HyperTypeValueEnum.INT,
        )
        self._default_hyper_parameters["subsample"] = HyperParametersDefinition(
            low=0,
            high=1,
            value=0,
            type_value=HyperTypeValueEnum.INT,
        )
        self._default_hyper_parameters["learning_rate"] = HyperParametersDefinition(
            low=0,
            high=1,
            value=1,
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._default_hyper_parameters["alpha"] = HyperParametersDefinition(
            low=0.01,
            high=1,
            value=0.01,
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._hyper_parameters = self._default_hyper_parameters
        self._default_hyper_parameters["lambda"] = HyperParametersDefinition(
            low=0.01,
            high=1,
            value=.01,
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._hyper_parameters = self._default_hyper_parameters

    def build_machine(self) -> None:

        self._machine = XGBRegressor(
            n_estimators=self._hyper_parameters["n_estimators"].value,
            eta=self._hyper_parameters["eta"].value,
            gamma=self._hyper_parameters["gamma"].value,
            max_depth=self._hyper_parameters["max_depth"].value,
            min_child_weight=self._hyper_parameters["min_child_weight"].value,
            max_delta_step=self._hyper_parameters["max_delta_step"].value,
            subsample=self._hyper_parameters["subsample"].value,
            alpha=self._default_hyper_parameters["alpha"].value,
            learning_rate=self._hyper_parameters["learning_rate"].value,
            reg_lambda=self._hyper_parameters["lambda"].value,
        )

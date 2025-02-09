"""To run machine to prediction"""

import random
import time
import json
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Union


from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.metrics.metric import Metric
from src.machines.machine_enums import MachineNames, HyperTypeValueEnum, LimitHyperParameter


class HyperParametersDefinition:
    """General definition for hyperparameters
    """

    def __init__(
        self,
        value: str,
        limit_hyper_parameter: LimitHyperParameter = LimitHyperParameter(),
        type_value: HyperTypeValueEnum = HyperTypeValueEnum.FLOAT,
    ):
        self._type_value = type_value
        self._value_str = value
        self._low_str = limit_hyper_parameter.low_value
        self._high_str = limit_hyper_parameter.high_value
        self._catalogue_values = limit_hyper_parameter.catalogue_values
        if self._low_str is None and self._catalogue_values is None:
            raise ValueError(f"Not valid hyper parameter with value - {value}")

    @property
    def type_value(self) -> HyperTypeValueEnum:
        """Return the type of hyper parameter"""
        return self._type_value

    def set_value(self, value: str) -> None:
        """Set a new value on str"""
        if self._type_value == HyperTypeValueEnum.INT:
            self._value_str = f"{int(value)}"
        if self._type_value == HyperTypeValueEnum.FLOAT:
            self._value_str = f"{float(value)}"
        if self._type_value == HyperTypeValueEnum.CATEGORY:
            if value not in self._catalogue_values:
                raise ValueError(f"Value: {value} is not on category")
            self._value_str = f"{value}"

    @property
    def value(self) -> Union[str, float, int]:
        """Return the cast to really value

        Raises:
            ValueError: Not value to return

        Returns:
            Union[str, float, int]: Value
        """
        if self._type_value == HyperTypeValueEnum.INT:
            return int(self._value_str)
        if self._type_value == HyperTypeValueEnum.FLOAT:
            return float(self._value_str)
        if self._type_value == HyperTypeValueEnum.CATEGORY:
            return self._value_str
        raise ValueError("Not value to return")

    def mutate_value(self, deep_decimal: int) -> None:
        """Mutate a value 

        Args:
            deep_decimal (int): deep of decimal to mutate

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if self._type_value == HyperTypeValueEnum.FLOAT:
            self.set_value(value=f"{round(random.uniform(
                float(self._low_str), float(self._high_str)),
                deep_decimal
            )}")
        if self._type_value == HyperTypeValueEnum.INT:
            self.set_value(
                value=f"{int(random.randint(int(self._low_str), int(self._high_str)))}")
        if self._type_value == HyperTypeValueEnum.CATEGORY:
            self.set_value(
                value=f"{random.choice(self._catalogue_values)}")


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
            self._hyper_parameters[key].mutate_value(
                deep_decimal=deep_decimal,
            )

    def mutate_hyper_parameters(self, mutation_rate: float, deep_decimal: int):
        """Mutate the hyper parameters

        Args:
            mutation_rate (float): Range between 0 and 1 to mutate the hyper parameters
        """
        for key, _ in self._hyper_parameters.items():
            if random.random() < mutation_rate:
                self._hyper_parameters[key].mutate_value(
                    deep_decimal=deep_decimal,
                )


class BayesianRegression(MachineRegression):
    """Class to run a Bayesian Prediction
    """

    def __init__(self) -> None:
        super().__init__()
        self._machine_name = MachineNames.BAYESIAN_REGRESSION.value
        self._default_hyper_parameters = {}
        self._hyper_parameters = {}
        self._default_hyper_parameters["alpha_1"] = HyperParametersDefinition(
            value="0.000001",
            limit_hyper_parameter=LimitHyperParameter(
                low_value="1e-6", high_value="1e-1"),
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._default_hyper_parameters["lambda_1"] = HyperParametersDefinition(
            value="0.000001",
            limit_hyper_parameter=LimitHyperParameter(
                low_value="1e-6", high_value="1e-1",),
            type_value=HyperTypeValueEnum.FLOAT,
        )
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


class RandomForestRegression(MachineRegression):
    """Machine for Random forest

    Args:
        MachinePrediction (_type_): Abstract method
    """

    def __init__(self) -> None:
        super().__init__()
        self._machine_name = MachineNames.RANDOM_FOREST_REGRESSION.value
        self._default_hyper_parameters = {}
        self._hyper_parameters = {}
        self._default_hyper_parameters["n_estimators"] = HyperParametersDefinition(
            value="100",
            limit_hyper_parameter=LimitHyperParameter(
                low_value="100", high_value="1000000"),
            type_value=HyperTypeValueEnum.INT,
        )
        self._default_hyper_parameters["min_samples_split"] = HyperParametersDefinition(
            value="0.1",
            limit_hyper_parameter=LimitHyperParameter(
                low_value="0.1", high_value="1",),
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._default_hyper_parameters["ccp_alpha"] = HyperParametersDefinition(
            value="0.1",
            limit_hyper_parameter=LimitHyperParameter(
                low_value="0.1", high_value="100",),
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._default_hyper_parameters["max_leaf_nodes"] = HyperParametersDefinition(
            value="100",
            limit_hyper_parameter=LimitHyperParameter(
                low_value="100", high_value="10000",),
            type_value=HyperTypeValueEnum.INT,
        )
        self._default_hyper_parameters["min_impurity_decrease"] = HyperParametersDefinition(
            value="0.0",
            limit_hyper_parameter=LimitHyperParameter(
                low_value="0.0", high_value="1",),
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._hyper_parameters = self._default_hyper_parameters

    def build_machine(self) -> None:
        self._machine = RandomForestRegressor(
            n_estimators=self._hyper_parameters["n_estimators"].value,
            min_samples_split=self._hyper_parameters["min_samples_split"].value,
            ccp_alpha=self._hyper_parameters["ccp_alpha"].value,
            max_leaf_nodes=self._hyper_parameters["max_leaf_nodes"].value,
            min_impurity_decrease=self._hyper_parameters["min_impurity_decrease"].value
        )


class SupportVectorRegression(MachineRegression):
    """Machine for Support Vector Regression

    Args:
        MachinePrediction (_type_): Abstract method
    """

    def __init__(self) -> None:
        super().__init__()
        self._machine_name = MachineNames.SUPPORT_VECTOR_REGRESSION.value
        self._default_hyper_parameters = {}
        self._hyper_parameters = {}
        self._default_hyper_parameters["kernel"] = HyperParametersDefinition(
            value="linear",
            limit_hyper_parameter=LimitHyperParameter(
                catalogue_values=[
                    'linear',
                    'poly',
                    'rbf',
                    'sigmoid'
                ],
            ),
            type_value=HyperTypeValueEnum.CATEGORY,
        )
        self._default_hyper_parameters["degree"] = HyperParametersDefinition(
            value="3",
            limit_hyper_parameter=LimitHyperParameter(
                low_value="1", high_value="10",),
            type_value=HyperTypeValueEnum.INT,
        )
        self._default_hyper_parameters["gama"] = HyperParametersDefinition(
            value="scale",
            limit_hyper_parameter=LimitHyperParameter(
                catalogue_values=[
                    'scale',
                    'auto'
                ],
            ),
            type_value=HyperTypeValueEnum.CATEGORY,
        )
        self._default_hyper_parameters["c"] = HyperParametersDefinition(
            value="1",
            limit_hyper_parameter=LimitHyperParameter(
                low_value="0.001",
                high_value="1",
            ),
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._default_hyper_parameters["epsilon"] = HyperParametersDefinition(
            value="0.1",
            limit_hyper_parameter=LimitHyperParameter(
                low_value="0.1",
                high_value="10",
            ),
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._default_hyper_parameters["coef0"] = HyperParametersDefinition(
            value="0.0",
            limit_hyper_parameter=LimitHyperParameter(
                low_value="0.001",
                high_value="0",
            ),
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._default_hyper_parameters["tol"] = HyperParametersDefinition(
            value="0.001",
            limit_hyper_parameter=LimitHyperParameter(
                low_value="0.0001",
                high_value="1.0",
            ),
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._hyper_parameters = self._default_hyper_parameters

    def build_machine(self) -> None:
        self._machine = SVR(
            kernel=self._hyper_parameters["kernel"].value,
            C=self._hyper_parameters["c"].value,
            epsilon=self._hyper_parameters["epsilon"].value,
            degree=self._hyper_parameters["degree"].value,
            gamma=self._hyper_parameters["gama"].value,
            coef0=self._hyper_parameters["coef0"].value,
            tol=self._hyper_parameters["tol"].value,
        )


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
            value="100",
            limit_hyper_parameter=LimitHyperParameter(
                low_value="10",
                high_value="10000",
            ),
            type_value=HyperTypeValueEnum.INT,
        )
        self._default_hyper_parameters["eta"] = HyperParametersDefinition(
            value=0.3,
            limit_hyper_parameter=LimitHyperParameter(
                low_value='0',
                high_value='1',
            ),
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._default_hyper_parameters["gamma"] = HyperParametersDefinition(
            value='0.3',
            limit_hyper_parameter=LimitHyperParameter(
                low_value='0',
                high_value='1',
            ),
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._default_hyper_parameters["max_depth"] = HyperParametersDefinition(
            limit_hyper_parameter=LimitHyperParameter(
                low_value='0',
                high_value='300',
            ),
            value='6',
            type_value=HyperTypeValueEnum.INT,
        )
        self._default_hyper_parameters["min_child_weight"] = HyperParametersDefinition(
            limit_hyper_parameter=LimitHyperParameter(
                low_value='0',
                high_value='300',
            ),
            value='1',
            type_value=HyperTypeValueEnum.INT,
        )
        self._default_hyper_parameters["max_delta_step"] = HyperParametersDefinition(
            limit_hyper_parameter=LimitHyperParameter(
                low_value='0',
                high_value='300',
            ),
            value='0',
            type_value=HyperTypeValueEnum.INT,
        )
        self._default_hyper_parameters["subsample"] = HyperParametersDefinition(
            limit_hyper_parameter=LimitHyperParameter(
                low_value='0',
                high_value='1',
            ),
            value='0',
            type_value=HyperTypeValueEnum.INT,
        )
        self._default_hyper_parameters["learning_rate"] = HyperParametersDefinition(
            limit_hyper_parameter=LimitHyperParameter(
                low_value='0',
                high_value='1',
            ),
            value='1',
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._default_hyper_parameters["alpha"] = HyperParametersDefinition(
            limit_hyper_parameter=LimitHyperParameter(
                low_value='0.01',
                high_value='1',
            ),
            value='0.01',
            type_value=HyperTypeValueEnum.FLOAT,
        )
        self._hyper_parameters = self._default_hyper_parameters
        self._default_hyper_parameters["lambda"] = HyperParametersDefinition(
            limit_hyper_parameter=LimitHyperParameter(
                low_value='0.01',
                high_value='1',
            ),
            value='0.01',
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

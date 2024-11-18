"""To run machine to prediction"""
import time
from datetime import timedelta
from abc import ABC, abstractmethod

from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from numpy import ndarray

from src.machines.machine_enums import SupportVectorKernelEnum
from src.metrics.error_metric_process import ErrorMetric
from src.machines.machine_enums import MachineJson, MachineNames


class MachineRegression(ABC):
    """Machine for regression and prediction
    """

    def __init__(self) -> None:
        self.x_train = None
        self.time_training = 0
        self.machine = None
        self.error_metric = None

    @abstractmethod
    def build_machine(self) -> None:
        """Create a machine for training and predict
        """

    def training(self, x_train: ndarray, y_train: ndarray) -> None:
        """Training function

        Args:
            x_train (ndarray): Vector for training
            y_train (ndarray): Vector to predict
        """
        start = time.time()
        self.machine.fit(x_train, y_train)
        self.time_training = timedelta(seconds=time.time() - start)

    def test(
        self,
        x_test: ndarray,
        y_test: ndarray,
    ) -> ErrorMetric:
        """Run test on machine and store data on error

        Args:
            x_test (ndarray): _description_
            y_test (ndarray): _description_
        """
        return ErrorMetric(
            x_test=x_test,
            y_test=y_test,
            y_predicted=self.machine.predict(x_test)
        )

        # def save_metric(
        #     self,
        #     x_test: ndarray,
        #     y_test,
        #     base_file_name: str,
        #     machine_name: str
        # ) -> None:
        #     """Save metric into 2 files all metrics and result of test

        #     Args:
        #         x_test (ndarray): Array for test
        #         y_test (_type_): Target for test
        #         base_file_name (str): path to save the other files
        #         machine_name (str): machine name

        #     Returns:
        #         _type_: _description_
        #     """
        #     print(f"Parameters: base file - {base_file_name}")

        #     self.error.calculate_metric_prediction()
        #     df_metric = self.error
        #     self.error.to_save(base_file_name=f"{machine_name}_{base_file_name}")

        # def get_metric(
        #     self,
        #     x_test: ndarray,
        #     y_test,
        # ) -> Dict:
        #     """Get the metric for some test data

        #     Args:
        #         x_test (ndarray): Array for test
        #         y_test (_type_): Target for test

        #     Returns:
        #         _type_: _description_
        #    """
        #     y_predicted = self.machine.predict(x_test)
        #     self.error = ErrorMetric(
        #         x_test=x_test, y_test=y_test, y_predicted=y_predicted)
        #     self.error.calculate_metric_prediction()
        #     return self.error.metrics

    def prediction(self, x_test: ndarray) -> ndarray:
        """Predict new values

        Args:
            x_test (ndarray): Vector for predict

        Returns:
            ndarray: Vector predicted
        """
        return self.machine.predict(x_test)


class BayesianRegression(MachineRegression):
    """Class to run a Bayesian Prediction
    """

    def build_machine(self) -> None:
        self.machine = BayesianRidge()

    def __str__(self) -> str:
        return f"Bayesian - {1} "


class LinearRRegression(MachineRegression):
    """Machine for linear regression
    """

    def build_machine(self) -> None:
        self.machine = LinearRegression()

    def __str__(self) -> str:
        return f"Linear regression - {1} "


class RidgeRegression(MachineRegression):
    """Machine for ridge regression

    Args:
        MachinePrediction (_type_): Abstract method
    """

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha

    def build_machine(self) -> None:
        self.machine = Ridge(alpha=self.alpha)

    def __str__(self) -> str:
        return f"Ridge - {self.alpha} "


class LassoRegression(MachineRegression):
    """Machine for linear LASSO

    Args:
        MachinePrediction (_type_): Abstract method
    """

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha

    def build_machine(self) -> None:
        self.machine = Lasso(alpha=self.alpha)

    def __str__(self) -> str:
        return f"LASSO - {self.alpha} "


class RandomForestRegression(MachineRegression):
    """Machine for Random forest

    Args:
        MachinePrediction (_type_): Abstract method 
    """

    def __init__(self, n_estimators: int = 1000, random_sate: int = 42, n_jobs: int = -1) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.random_sate = random_sate
        self.n_jobs = n_jobs

    def build_machine(self) -> None:
        self.machine = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_sate,
            n_jobs=self.n_jobs,
        )

    def __str__(self) -> str:
        return f"Random Forest Prediction - {self.n_estimators} "


class SupportVectorRegression(MachineRegression):
    """Machine for Support Vector Regression

    Args:
        MachinePrediction (_type_): Abstract method
    """

    def build_machine(self) -> None:
        self.machine = SVR(
            kernel=self.kernel.value,
            C=self.c,
            epsilon=self.epsilon,
        )

    def __str__(self) -> str:
        return f"Support Vector Regression - {self.kernel.value}"

    def __init__(
        self,
        kernel: SupportVectorKernelEnum = SupportVectorKernelEnum.LINEAR,
        c: float = 1.0,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.c = c
        self.epsilon = epsilon


class ExtremeGradientBoostRegression(MachineRegression):
    """Machine for prediction on Extreme Gradient Boosting

    Args:
        MachinePrediction (_type_): Abstract method
    """

    def __init__(self, n_estimators: int = 1000, max_depth: int = -1, max_leaves: int = 0) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaves = max_leaves

    def build_machine(self) -> None:
        self.machine = XGBRegressor(
            # n_estimators=self.n_estimators,
            # max_depth=self.max_depth,
            # max_leaves=self.max_leaves,
        )

    def __str__(self) -> str:
        return f"Extreme Gradient Boost - {self.n_estimators} "


def machine_build_regression(machine_definition: MachineJson) -> MachineRegression:
    """Build some machine json 

    Args:
        machine_definition (MachineJson): Machine definition

    Raises:
        NotImplementedError: The machine don't exist

    Returns:
        MachineJson: Return child of Machine Json to run it
    """
    if machine_definition.name == MachineNames.RF:
        return RandomForestRegression(
            n_estimators=machine_definition.n_estimators,
            random_sate=machine_definition.random_state,
            n_jobs=machine_definition.n_jobs,
        )
    if machine_definition.name == MachineNames.XGB:
        return ExtremeGradientBoostRegression()
    if machine_definition.name == MachineNames.BAP:
        return BayesianRegression()
    if machine_definition.name == MachineNames.LAP:
        return LassoRegression()
    if machine_definition.name == MachineNames.SVRP:
        return SupportVectorRegression()
    raise NotImplementedError("Don't exist machine to run it")

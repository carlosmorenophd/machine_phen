"""To run machine to prediction"""
import time
from datetime import timedelta
from abc import ABC, abstractmethod

from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from numpy import ndarray

from src.machines.enums import SvrKernelEnum
from src.metrics.error_metric import ErrorMetric


class MachinePrediction(ABC):
    """Machine for predict
    """

    def __init__(self) -> None:
        self.x_train = None
        self.time_training = 0
        self.machine = None
        self.error = None

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

    def save_metric(
        self, x_test: ndarray, y_test, base_file_name: str
    ) -> None:
        """Save metric into 2 files all metrics and result of test

        Args:
            x_test (ndarray): Array for test
            y_test (_type_): Target for test
            file_metric_name (str): file to save the metric
            file_result_name (str): file to save the target, predict and error

        Returns:
            _type_: _description_
        """
        y_predicted = self.machine.predict(x_test)
        self.error = ErrorMetric(x_test=x_test, y_test=y_test, y_predicted=y_predicted)
        self.error.calculate_metric_prediction()
        self.error.to_save(base_file_name=base_file_name)





    def prediction(self, x_test: ndarray) -> ndarray:
        """Predict new values

        Args:
            x_test (ndarray): Vector for predict

        Returns:
            ndarray: Vector predicted
        """
        return self.machine.predict(x_test)


# class BayesianPrediction(MachinePrediction):
#     """Class to run a Bayesian Prediction
#     """

#     def training(self, x_train: ndarray, y_train: ndarray) -> None:
#         self.x_train = x_train
#         self.machine = BayesianRidge()
#         start = time.time()
#         self.machine.fit(X=x_train, y=y_train)
#         self.time_training = timedelta(seconds=time.time() - start)

#     def prediction(self, x_test: ndarray) -> ndarray:
#         return self.machine.predict(X=x_test)


# class LinearRegressionPrediction(MachinePrediction):
#     """Machine for linear regression

#     Args:
#         MachinePrediction (_type_): Abstract method
#     """

#     def training(self, x_train: ndarray, y_train: ndarray) -> None:
#         self.x_train = x_train
#         self.machine = LinearRegression()
#         start = time.time()
#         self.machine.fit(X=x_train, y=y_train)
#         self.time_training = timedelta(seconds=time.time() - start)

#     def prediction(self, x_test: ndarray) -> ndarray:
#         return self.machine.predict(X=x_test)


# class RidgePrediction(MachinePrediction):
#     """Machine for ridge regression

#     Args:
#         MachinePrediction (_type_): Abstract method
#     """

#     def __init__(self, alpha: float = 0.1) -> None:
#         super().__init__()
#         self.machine = Ridge(alpha=alpha)

#     def training(self, x_train: ndarray, y_train: ndarray) -> None:
#         self.x_train = x_train
#         start = time.time()
#         self.machine.fit(X=x_train, y=y_train)
#         self.time_training = timedelta(seconds=time.time() - start)

#     def prediction(self, x_test: ndarray) -> ndarray:
#         return self.machine.predict(X=x_test)


# class LassoPrediction(MachinePrediction):
#     """Machine for linear LASSO

#     Args:
#         MachinePrediction (_type_): Abstract method
#     """

#     def __init__(self, alpha: float = 0.1) -> None:
#         super().__init__()
#         self.alpha = alpha

#     def training(self, x_train: ndarray, y_train: ndarray) -> None:
#         self.x_train = x_train
#         self.machine = Lasso(alpha=self.alpha)
#         start = time.time()
#         self.machine.fit(X=x_train, y=y_train)
#         self.time_training = timedelta(seconds=time.time() - start)

#     def prediction(self, x_test: ndarray) -> ndarray:
#         return self.machine.predict(X=x_test)


class RandomForestPrediction(MachinePrediction):
    """Machine for Random forest

    Args:
        MachinePrediction (_type_): Abstract method 
    """

    def __init__(self, n_estimators: int = 1000, random_sate: int = 42, n_jobs: int = -1) -> None:
        super().__init__()
        self.n_estimators = n_estimators if n_estimators > 0 else 1000
        self.random_sate = random_sate if random_sate > 0 else 42
        self.n_jobs = n_jobs if n_jobs > -1 else -1

    def build_machine(self) -> None:
        self.machine = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_sate,
            n_jobs=self.n_jobs,
        )

    def __str__(self) -> str:
        return f"Random Forest Prediction - {self.n_estimators} "


# class SupportVectorRegressionPrediction(MachinePrediction):
#     """Machine for Support Vector Regression

#     Args:
#         MachinePrediction (_type_): Abstract method
#     """

#     def __init__(
#         self,
#         kernel: SvrKernelEnum = SvrKernelEnum.RBF,
#         c: float = 1.0,
#         epsilon: float = 0.1,
#     ) -> None:
#         super().__init__()
#         self.kernel = kernel
#         self.c = c
#         self.epsilon = epsilon

#     def training(self, x_train: ndarray, y_train: ndarray) -> None:
#         self.machine = SVR(
#             kernel=self.kernel.value,
#             C=self.c,
#             epsilon=self.epsilon,
#         )
#         start = time.time()
#         self.machine.fit(x_train, y_train)
#         self.time_training = timedelta(seconds=time.time() - start)

#     def prediction(self, x_test: ndarray) -> ndarray:
#         return self.machine.predict(x_test)

#     def __str__(self) -> str:
#         return f"Support Vector Regression - {self.kernel.value}"


# class ExtremeGradientBoostingPrediction(MachinePrediction):
#     """Machine for prediction on Extreme Gradient Boosting

#     Args:
#         MachinePrediction (_type_): Abstract method
#     """

#     def training(self, x_train: ndarray, y_train: ndarray) -> None:
#         self.machine = XGBRegressor()
#         start = time.time()
#         self.machine.fit(x_train, y_train)
#         self.time_training = timedelta(seconds=time.time() - start)

#     def prediction(self, x_test: ndarray) -> ndarray:
#         return self.machine.predict(x_test)

#     def __str__(self) -> str:
#         return f"Extreme Gradient Boosting - {1}"

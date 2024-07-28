from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from numpy import ndarray
from machines.enums import SvrKernelEnum
import time
from datetime import timedelta


class BayesianPrediction:
    def __init__(self) -> None:
        self.x_train = None
        self.time_training = 0

    def training(self, x_train: ndarray, y_train: ndarray) -> None:
        self.x_train = x_train
        self.bayesian = BayesianRidge()
        self.bayesian.fit(X=x_train, y=y_train)
        print("Score -> {}".format(self.bayesian.score(X=x_train, y=y_train)))

    def prediction(self, x_test: ndarray) -> ndarray:
        return self.bayesian.predict(X=x_test)


class LinearRegressionPrediction:
    def __init__(self) -> None:
        self.linear = None
        self.time_training = 0

    def training(self, x_train: ndarray, y_train: ndarray) -> None:
        self.x_train = x_train
        self.linear = LinearRegression()
        self.linear.fit(X=x_train, y=y_train)

    def prediction(self, x_test: ndarray) -> ndarray:
        return self.linear.predict(X=x_test)


class RidgePrediction:
    def __init__(self) -> None:
        self.ridge = None
        self.time_training = 0

    def training(self, x_train: ndarray, y_train: ndarray, alpha: float = 0.1) -> None:
        self.x_train = x_train
        self.ridge = Ridge(alpha=alpha)
        self.ridge.fit(X=x_train, y=y_train)

    def prediction(self, x_test: ndarray) -> ndarray:
        return self.ridge.predict(X=x_test)


class LassoPrediction:
    def __init__(self) -> None:
        self.lasso = None
        self.time_training = 0

    def training(self, x_train: ndarray, y_train: ndarray, alpha: float = 0.1) -> None:
        self.x_train = x_train
        self.lasso = Lasso(alpha=alpha)
        self.lasso.fit(X=x_train, y=y_train)

    def prediction(self, x_test: ndarray) -> ndarray:
        return self.lasso.predict(X=x_test)


class RF_Prediction:
    def __init__(self, n_estimators=1000, random_sate=42, n_jobs: int = -1) -> None:
        self.n_estimators = n_estimators
        self.random_sate = random_sate
        self.n_jobs = n_jobs
        self.time_training = 0

    def training(self, x_train: ndarray, y_train: ndarray) -> None:
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_sate,
            n_jobs=self.n_jobs,
        )
        self.rf.fit(x_train, y_train)

    def prediction(self, x_test: ndarray) -> ndarray:
        return self.rf.predict(x_test)


class SVR_Prediction:
    def __init__(
        self,
        kernel: SvrKernelEnum = SvrKernelEnum.RBF,
        c: float = 1.0,
        epsilon: float = 0.1,
    ) -> None:
        self.kernel = kernel
        self.c = c
        self.epsilon = epsilon
        self.time_training = 0

    def training(self, x_train: ndarray, y_train: ndarray) -> None:
        self.svr = SVR(
            kernel=self.kernel.value,
            C=self.c,
            epsilon=self.epsilon,
        )
        start = time.time()
        self.svr.fit(x_train, y_train)
        end = time.time()
        self.time_training = timedelta(seconds=end - start)

    def prediction(self, x_test: ndarray) -> ndarray:
        return self.svr.predict(x_test)

    def __str__(self) -> str:
        return f"SVR - {self.kernel.value}"


class XGB_Prediction:
    def __init__(
        self,
        is_debug: bool = False,
    ) -> None:
        self.is_debug = is_debug
        self.time_training = 0

    def training(self, x_train: ndarray, y_train: ndarray) -> None:
        self.xgb = XGBRegressor()
        self.xgb.fit(x_train, y_train)

    def prediction(self, x_test: ndarray) -> ndarray:
        return self.xgb.predict(x_test)

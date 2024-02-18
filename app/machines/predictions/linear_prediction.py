from sklearn.linear_model import LinearRegression, Ridge, Lasso
from numpy import ndarray


class LinearRegressionPrediction:
    def __init__(self) -> None:
        self.linear = None

    def training(self, x_train: ndarray, y_train: ndarray) -> None:
        self.x_train = x_train
        self.linear = LinearRegression()
        self.linear.fit(X=x_train, y=y_train)

    def prediction(self, x_test: ndarray) -> ndarray:
        return self.linear.predict(X=x_test)


class RidgePrediction:
    def __init__(self) -> None:
        self.ridge = None

    def training(self, x_train: ndarray, y_train: ndarray, alpha: float = 0.1) -> None:
        self.x_train = x_train
        self.ridge = Ridge(alpha=alpha)
        self.ridge.fit(X=x_train, y=y_train)

    def prediction(self, x_test: ndarray) -> ndarray:
        return self.ridge.predict(X=x_test)


class LassoPrediction:
    def __init__(self) -> None:
        self.lasso = None

    def training(self, x_train: ndarray, y_train: ndarray, alpha: float = 0.1) -> None:
        self.x_train = x_train
        self.lasso = Lasso(alpha=alpha)
        self.lasso.fit(X=x_train, y=y_train)

    def prediction(self, x_test: ndarray) -> ndarray:
        return self.lasso.predict(X=x_test)

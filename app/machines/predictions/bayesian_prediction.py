from sklearn.linear_model import BayesianRidge
from numpy import ndarray


class BayesianPrediction:
    def __init__(self) -> None:
        self.x_train = None

    def training(self, x_train: ndarray, y_train: ndarray) -> None:
        self.x_train = x_train
        self.bayesian = BayesianRidge()
        self.bayesian.fit(X=x_train, y=y_train)

    def prediction(self, x_test: ndarray) -> ndarray:
        return self.bayesian.predict(X=x_test)

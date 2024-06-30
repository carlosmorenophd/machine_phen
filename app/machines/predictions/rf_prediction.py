from sklearn.ensemble import RandomForestRegressor
from machines.enums import SvrKernelEnum
from numpy import ndarray


class RF_Prediction:
    def __init__(self, n_estimators=1000, random_sate=42) -> None:
        self.n_estimators = n_estimators
        self.random_sate = random_sate

    def training(self, x_train: ndarray, y_train: ndarray) -> None:
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_sate
        )
        self.rf.fit(x_train, y_train)

    def prediction(self, x_test: ndarray) -> ndarray:
        return self.rf.predict(x_test)

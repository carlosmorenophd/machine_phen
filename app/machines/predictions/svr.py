
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from machines.enums import SvrKernelEnum
from numpy import ndarray


class SVR_Prediction():
    def __init__(self, kernel: SvrKernelEnum = SvrKernelEnum.RBF, c: float = 1.0, epsilon: float = 0.1) -> None:
        self.kernel = kernel
        self.c = c
        self.epsilon = epsilon
        self.precision = None

    def training(self, x_train: ndarray, y_train: ndarray) -> None:
        self.svr = SVR(
            kernel=self.kernel.value,
            C=self.c,
            epsilon=self.epsilon,
        )
        self.svr.fit(x_train, y_train)

    def prediction(self, x_test: ndarray) -> ndarray:
        return self.svr.predict(x_test)

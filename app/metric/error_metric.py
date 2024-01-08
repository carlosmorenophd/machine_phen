from sklearn.metrics import mean_squared_error


class CalculateError():
    def __init__(self, predictions, y_test) -> None:
        self.predictions = predictions
        self.y_test = y_test

    def get_mean_squared_error(self):
        return mean_squared_error(self.y_test, self.predictions)

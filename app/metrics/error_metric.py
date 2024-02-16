from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from numpy import ndarray

class ErrorMetric():
    def __init__(self, y_predicted, y_test) -> None:
        self.y_predicted = y_predicted
        self.y_test = y_test
        self.mean_squared_error = None
        self.determination_coefficient = None
        self.mean_absolute_percentage_error = None
    
    def calculate_metric_prediction(self) -> None:
        self.mean_squared_error = mean_squared_error(y_pred=self.y_predicted, y_true=self.y_test)
        self.determination_coefficient = r2_score(y_pred=self.y_predicted, y_true=self.y_test)
        self.mean_absolute_percentage_error = mean_absolute_percentage_error(y_pred=self.y_predicted, y_true=self.y_test)

    def print_list_on_error_upper(self, pivot: float, debug: bool = False) -> ndarray:
        number_upper = 0
        list_upper = []
        if debug:
            print("#### list on error upper -> {}".format(pivot))
        for i in range(len(self.y_predicted)):
            error = abs(self.y_predicted[i] - self.y_test[i])
            if error > pivot:
                number_upper = number_upper + 1
                list_upper.append({'truth': self.y_test[i], 'predict': self.y_predicted[i], 'error': error})
                if debug:
                    print('Predict {} -> Truth {} dif = {}'.format(self.y_predicted[i], self.y_test[i], error))
        if debug:
            print("Number of element -> {} number of upper error -> {}".format(len(self.y_test), number_upper))
        return list_upper

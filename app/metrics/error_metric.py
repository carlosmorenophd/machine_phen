from sklearn.metrics import (
    d2_absolute_error_score,
    d2_pinball_score,
    d2_tweedie_score,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
from numpy import ndarray
import matplotlib.pyplot as plt
from metrics.enums import MetricEnum, PlotLegends


class ErrorMetric():
    def __init__(self, y_predicted, y_test, x_test) -> None:
        self.y_predicted = y_predicted
        self.y_true = y_test
        self.x_true = x_test
        self.metrics = {}

    def plot_true_vs_predicted(
        self,
        legend: PlotLegends = PlotLegends(),
        is_inline: bool = True,
        save_file: str = "",
    ):
        plt.scatter(
            self.y_true,
            self.y_predicted,
            label=legend.first_plot_label,
        )
        plt.plot(
            [min(self.y_true), max(self.y_true)], [
                min(self.y_true), max(self.y_true)],
            linestyle='--',
            color='red',
            linewidth=2,
            label=legend.second_plot_label,
        )
        plt.xlabel(legend.x_label)
        plt.ylabel(legend.y_label)
        plt.legend()
        plt.title(legend.title)
        if is_inline:
            plt.show()
        if not save_file == "":
            plt.savefig(save_file)

    def plot_r2_predicted(
        self,
        legend: PlotLegends = PlotLegends(),
        is_inline: bool = True,
    ):
        for x_true_single in self.x_true.T:
            plt.scatter(
                x_true_single,
                self.y_true,
                color='red',
                label=legend.first_plot_label,
            )
            plt.plot(
                x_true_single,
                self.y_predicted,
                label=legend.second_plot_label
            )
            plt.xlabel(legend.x_label)
            plt.ylabel(legend.y_label)
            plt.legend()
            plt.title(legend.title)
            if is_inline:
                plt.show()
            break

    def calculate_metric_prediction(self) -> None:
        self.metrics[MetricEnum.D2_ABSOLUTE_ERROR_SCORE.value] = d2_absolute_error_score(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.D2_PINBALL_SCORE.value] = d2_pinball_score(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.D2_TWEEDIE_SCORE.value] = d2_tweedie_score(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.EXPLAINED_VARIANCE_SCORE.value] = explained_variance_score(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.MAX_ERROR.value] = max_error(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.MEAN_ABSOLUTE_ERROR.value] = mean_absolute_error(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR.value] = mean_absolute_percentage_error(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.MEAN_GAMMA_DEVIANCE.value] = mean_gamma_deviance(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.MEAN_POISSON_DEVIANCE.value] = mean_poisson_deviance(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.MEAN_SQUARED_ERROR.value] = mean_squared_error(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.MEAN_SQUARED_LOG_ERROR.value] = mean_squared_log_error(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.MEDIAN_ABSOLUTE_ERROR.value] = median_absolute_error(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.R2_SCORE.value] = r2_score(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.ROOT_MEAN_SQUARED_ERROR.value] = root_mean_squared_error(
            y_pred=self.y_predicted, y_true=self.y_true)
        self.metrics[MetricEnum.ROOT_MEAN_SQUARED_LOG_ERROR.value] = root_mean_squared_log_error(
            y_pred=self.y_predicted, y_true=self.y_true)

    def get_metric(self, metric: MetricEnum) -> float:
        if metric.value in self.metrics.keys():
            return self.metrics[metric.value]
        else:
            raise Exception("Metric is not valid")

    def print_list_on_error_upper(self, pivot: float, debug: bool = False) -> ndarray:
        number_upper = 0
        list_upper = []
        if debug:
            print("#### list on error upper -> {}".format(pivot))
        for i in range(len(self.y_predicted)):
            error = abs(self.y_predicted[i] - self.y_true[i])
            if error > pivot:
                number_upper = number_upper + 1
                list_upper.append(
                    {'truth': self.y_true[i], 'predict': self.y_predicted[i], 'error': error})
                if debug:
                    print(
                        'Predict {} -> Truth {} dif = {}'.format(self.y_predicted[i], self.y_true[i], error))
        if debug:
            print(
                "Number of element -> {} number of upper error -> {}".format(len(self.y_true), number_upper))
        return list_upper

    def get_all_from_predict(self) -> ndarray:
        return self.metrics

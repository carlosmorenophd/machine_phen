"""Get metric from machines    """
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
import pandas as pd
import matplotlib.pyplot as plt

from src.metrics.enums import MetricEnum, PlotLegends
from src.helpers.key_env import IS_DEBUG
from src.helpers.file_access import save_to_csv


class ErrorMetric():
    """Class to get error metrics
    """

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
        """Plot true vs predict

        Args:
            legend (PlotLegends, optional): Is the basic attributes to draw a plot. Defaults to PlotLegends().
            is_inline (bool, optional): if return the graphic. Defaults to True.
            save_file (str, optional): save graphic on file. Defaults to "".
        """
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
        if save_file != "":
            plt.savefig(save_file)

    def plot_r2_predicted(
        self,
        features: ndarray,
        legend: PlotLegends = PlotLegends(),
        is_inline: bool = True,
        save_file: str = "",
    ):
        """Plot the r2 on predict

        Args:
            features (ndarray): list of features
            legend (PlotLegends, optional): basic legends for plot. Defaults to PlotLegends().
            is_inline (bool, optional): Plot in line. Defaults to True.
        """
        index_feature = 0
        for x_true_single in self.x_true.T:
            variable_name = features[index_feature]
            plt.scatter(
                x_true_single,
                self.y_true,
                color='red',
                label=legend.first_plot_label.format(variable=variable_name),
            )
            plt.scatter(
                x_true_single,
                self.y_predicted,
                label=legend.second_plot_label.format(variable=variable_name)
            )
            plt.xlabel(legend.x_label.format(variable=variable_name))
            plt.ylabel(legend.y_label)
            plt.legend()
            plt.title(legend.title.format(variable=variable_name,
                      r2=self.get_metric(metric=MetricEnum.R2_SCORE)))
            if is_inline:
                plt.show()
            index_feature = index_feature + 1
        if save_file != "":
            plt.savefig(save_file)

    def calculate_metric_prediction(self) -> None:
        """Calculate all metrics
        """
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
        self.metrics[MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR.value] =
        mean_absolute_percentage_error(
            y_pred=self.y_predicted, y_true=self.y_true
        )
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
        """Get metric by enum

        Args:
            metric (MetricEnum): Metric to get

        Raises:
            ModuleNotFoundError: Not found the metric

        Returns:
            float: return the metric
        """
        if metric.value in self.metrics.keys():
            return self.metrics[metric.value]
        raise ModuleNotFoundError("Metric is not valid")

    def print_list_on_error_upper(self, pivot: float) -> ndarray:
        """Print list of error

        Args:
            pivot (float): point to print

        Returns:
            ndarray: list od error
        """
        number_upper = 0
        list_upper = []
        if IS_DEBUG:
            print(f"#### list on error upper -> {pivot}")
        for i, predicted in enumerate(self.y_predicted):
            error = abs(predicted - self.y_true[i])
            if error > pivot:
                number_upper = number_upper + 1
                list_upper.append(
                    {'truth': self.y_true[i], 'predict': predicted, 'error': error})
                if IS_DEBUG:
                    print(
                        f'Predict {predicted} -> Truth {self.y_true[i]} dif = {error}')
        if IS_DEBUG:
            print(
                f"Number of element -> {self.y_true} number of upper error -> {number_upper}")
        return list_upper

    def list_percentage_error_upper(
        self,
        pivot: float,
        sort: str = 'error',
    ) -> ndarray:
        """List of error with upper first

        Args:
            pivot (float): Pivot to print
            sort (str, optional): Way to sorter. Defaults to 'error'.

        Returns:
            ndarray: _description_
        """
        number_upper = 0
        list_upper = []
        if IS_DEBUG:
            print(f"#### list on error upper -> {pivot}")
        for i, predicted in enumerate(self.y_predicted):
            error = abs(
                (self.y_true[i] - predicted) / self.y_true[i])
            if error > pivot:
                number_upper = number_upper + 1
                list_upper.append(
                    {'truth': self.y_true[i], 'predict': predicted, 'error': error})
                if IS_DEBUG:
                    print(
                        f'Predict {predicted} -> Truth {self.y_true[i]} dif = {error}')
        if IS_DEBUG:
            print(
                f"Number of element -> {len(self.y_true)} number of upper error -> {number_upper}")
        return sorted(list_upper, key=lambda x: x[sort], reverse=True)

    def get_all_from_predict(self) -> ndarray:
        """Get all metrics

        Returns:
            ndarray: List of metrics
        """
        return self.metrics

    def to_save(self, base_file_name: str) -> None:
        """Save all metric on files

        Args:
            base_file_name (str): Base path to save
        """
        file_metric = f"metric_{base_file_name}.csv"
        # file_result = f"result_{base_file_name}.csv"
        df_metric = pd.DataFrame(self.metrics)
        save_to_csv(data_frame=df_metric, file_save=file_metric)

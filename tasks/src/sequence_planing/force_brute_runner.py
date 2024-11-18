"""Runner to get best parameter with force brute """
from typing import Dict, List
import itertools

import pandas as pd
from src.machines.machine_enums import MachineJson, FileAccessRunnerProperties
from src.machines.machine_data_frame_handler import DataFrameHandler
from src.machines.regression_run import machine_build_regression
from src.metrics.metric_enums import MetricEnum


class ForwardSelectionBestResult():
    """Basic stores for forward selection best columns and other
    """

    def __init__(self, metric_to_evaluate: MetricEnum):
        columns = ["machine_name", "columns"]
        for metric in MetricEnum:
            columns.append(metric.value)
        self.metric_values = pd.DataFrame(columns=columns)
        self.metric_to_evaluate = metric_to_evaluate
        self.best_metric = -1
        self.best_machine = None
        if metric_to_evaluate == MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR:
            self.best_metric = 1000

    def add_metric_value(
            self,
            machine_definition: MachineJson,
            columns: List[str],
            metrics: Dict,
    ) -> None:
        """Add new metric and validate if is better that previous

        Args:
            machine (MachineJson): machine now
            metrics (Dict): metric for the machine
        """
        element = {
            "machine_name": str(machine_definition),
            "columns": ",".join(columns),
        }
        element = element | metrics
        self.metric_values = self.metric_values.append(
            element,
            ignore_index=True,
        )
        if self.metric_to_evaluate == MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR:
            self.validate_minus(
                value=metrics[self.metric_to_evaluate],
                machine=machine_definition
            )

    def validate_minus(self, value: float, machine: MachineJson) -> None:
        """Validate if is better by minus values

        Args:
            value (float): new value
            machine (MachineJson): new machine
        """
        if self.best_metric > value:
            self.best_metric = value
            self.best_machine = machine


def combination_columns_from_data_frame(df: pd.DataFrame, minimal: int = 1):
    """Combine all columns with min of columns on combination

    Args:
        df (pd.DataFrame): data frame to get the columns
        min (int, optional): minimal of column on group. Defaults to 1.

    Yields:
        _type_: return array of combination
    """
    columns = list(df.columns)
    for r in range(minimal, len(columns) + 1):
        for comb in itertools.combinations(columns, r):
            yield list(comb)


def forward_selection_force_brute_run(
    file_access_runner: FileAccessRunnerProperties,
    machine_definition: MachineJson,
    metric: MetricEnum = MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
) -> None:
    """Run forward selection with force brute
    Args:
            initial_column (_type_): _description_
            metric (MetricEnum, optional): metric to validate machine.
                Defaults to MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR.
    """
    data_frame = DataFrameHandler(file_access_runner=file_access_runner)
    forward_selection = ForwardSelectionBestResult(metric_to_evaluate=metric)
    df = data_frame.get_data_frame_without_target()
    machine = machine_build_regression(machine_definition=machine_definition)
    for combination in combination_columns_from_data_frame(df):
        data_frame.change_column_from_data_frame(columns_to_keep=combination)
        dataset = data_frame.dataset
        machine.build_machine()
        machine.training(dataset.x_train, dataset.y_train)
        error_metric = machine.test(
            x_test=dataset.x_test,
            y_test=dataset.y_test,
        )
        forward_selection.add_metric_value(
            machine_definition=machine_definition,
            columns=combination,
            metrics=error_metric.metrics,
        )
        df_metric = forward_selection.metric_values
        data_frame.storage_file.save_data_frame_to_csv(
            data_frame=df_metric, prefix="metric_forward_selection",
        )

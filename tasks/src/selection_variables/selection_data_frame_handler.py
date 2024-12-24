"""helper to work with data frame for selection the variable  """
from typing import Dict, List

import pandas as pd
from src.machines.machine_enums import MachineJson
from src.metrics.metric_enums import MetricEnum


class SelectionBestMetric():
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

    def exist_this_columns(
            self,
            machine_definition: MachineJson,
            columns: List[str],
    ):
        """Validate if was testing this combination of columns

        Args:
            columns (List[str]): list of columns to search
        """
        value = self.metric_values[
            (
                self.metric_values['machine_name'] == str(machine_definition)
            ) & (
                self.metric_values['columns'] == ",".join(columns)
            )]
        if value.empty:
            return None
        print(value[self.metric_to_evaluate.value].iloc[0])
        return value[self.metric_to_evaluate.value].iloc[0]

    def add_metric_value(
            self,
            machine_definition: MachineJson,
            columns: List[str],
            metrics: Dict,
    ) -> float:
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
        self.metric_values = pd.concat([
            self.metric_values,
            pd.DataFrame([element]),
        ], ignore_index=True)
        if self.metric_to_evaluate == MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR:
            self.validate_minus(
                value=metrics[self.metric_to_evaluate.value],
                machine=machine_definition
            )
            return metrics[self.metric_to_evaluate.value]
        raise KeyError("Not valid metric")

    def validate_minus(self, value: float, machine: MachineJson) -> None:
        """Validate if is better by minus values

        Args:
            value (float): new value
            machine (MachineJson): new machine
        """
        if self.best_metric > value:
            self.best_metric = value
            self.best_machine = machine

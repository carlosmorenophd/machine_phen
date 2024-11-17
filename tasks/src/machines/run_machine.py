"""Run machine"""

from dataclasses import dataclass
from typing import Dict
import itertools

from sklearn.model_selection import train_test_split
from numpy import ndarray
import pandas as pd
from src.machines.predictions import (
    RandomForestPrediction,
    ExtremeGradientBoostPrediction,
    BayesianPrediction,
    LassoPrediction,
    SupportVectorRegressionPrediction,
    MachinePrediction,
)
from machines.machine_enums import MachineNames, MachineJson
from src.helpers.file_access import FolderCache, get_file_to_data_frame
from src.metrics.metric_enums import MetricEnum


@dataclass
class FileAccessRunnerProperties:
    """Minimal parameter to load, split  the file and the target to run machine
    """
    target_feature: str
    file_in: str
    file_name_only: str
    folder_path: FolderCache = FolderCache.UPLOAD
    test_size: float = 0.8
    random_state: int = 42


@dataclass
class DatasetProperties:
    """All properties for get the file and pass to the machine
    """
    df: pd.DataFrame
    x: ndarray
    y: ndarray
    x_train: ndarray
    x_test: ndarray
    y_train: ndarray
    y_test: ndarray


class MachineRunner():
    """Run all machines for Predicted
    """

    def __init__(
        self,
        file_access_runner: FileAccessRunnerProperties,
    ) -> None:
        self.file_access_runner = file_access_runner
        df = get_file_to_data_frame(
            file_name=file_access_runner.file_in,
            folder=file_access_runner.folder_path
        )
        x = df.drop(
            file_access_runner.target_feature, axis=1)
        y = df[file_access_runner.target_feature]
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=file_access_runner.test_size,
            random_state=file_access_runner.random_state
        )
        self.dataset = DatasetProperties(
            df=df, x=x, y=y, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        self.columns_to_adding = []
        self.best_columns_df = []
        self.original_df = df

    def change_column_from_file(self, columns_to_keep: list):
        """Get some column of the dataset

        Args:
            columns_to_adding (list): list that columns to keep
        """
        self.columns_to_adding = columns_to_keep
        df = get_file_to_data_frame(
            file_name=self.file_access_runner.file_in,
            folder=self.file_access_runner.folder_path
        )
        x = df.drop(
            self.file_access_runner.target_feature, axis=1)
        x = x[columns_to_keep]
        y = df[self.file_access_runner.target_feature]
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.file_access_runner.test_size,
            random_state=self.file_access_runner.random_state
        )
        self.dataset = DatasetProperties(
            df=df,
            x=x,
            y=y,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

    def run_single_machine_single_file(
        self,
        machine_definition: MachineJson,
    ) -> None:
        """Run single machine with single file

        Args:
            machine (MachineJson): machine to run 
        """
        print(f"Parameters machine- {machine_definition}")
        machine = self.build_machine(machine_definition=machine_definition)
        if machine is not None:
            machine.build_machine()
            machine.training(self.dataset.x_train, self.dataset.y_train)
            machine.save_metric(
                x_test=self.dataset.x_test,
                y_test=self.dataset.y_test,
                base_file_name=self.file_access_runner.file_name_only,
                machine_name=machine_definition.name.value,
            )

    def build_machine(self, machine_definition: MachineJson) -> MachinePrediction:
        """Build some machine json 

        Args:
            machine_definition (MachineJson): Machine definition

        Returns:
            MachineJson: Return child of Machine Json to run it
        """
        if machine_definition.name == MachineNames.RF:
            return RandomForestPrediction(
                n_estimators=machine_definition.n_estimators,
                random_sate=machine_definition.random_state,
                n_jobs=machine_definition.n_jobs,
            )
        if machine_definition.name == MachineNames.XGB:
            return ExtremeGradientBoostPrediction()
        if machine_definition.name == MachineNames.BAP:
            return BayesianPrediction()
        if machine_definition.name == MachineNames.LAP:
            return LassoPrediction()
        if machine_definition.name == MachineNames.SVRP:
            return SupportVectorRegressionPrediction()
        return None

    def forward_selection_force_brute(
            self,
            machine_definition: MachineJson,
            metric: MetricEnum = MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
    ) -> None:
        """Get the best combination of variable by force brute on some machine

        Args:
            initial_column (_type_): _description_
            metric (MetricEnum, optional): _description_. Defaults to MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR.
        """
        forwardSelection = ForwardSelectionBest(metric_to_evaluate=metric)
        df = self.original_df.drop(
            self.file_access_runner.target_feature,
            axis=1,
        )
        for combination in combination_columns_from_data_frame(df):
            machine = self.build_machine(machine_definition=machine_definition)
            self.change_column_from_file(columns_to_keep=combination)
            machine.build_machine()
            machine.training(self.dataset.x_train, self.dataset.y_train)
            result = machine.get_metric(x_test=self.dataset.x_test,
                               y_test=self.dataset.y_test,)


class ForwardSelectionBest():
    """Basic stores for forward selection best columns and other
    """

    def __init__(self, metric_to_evaluate: MetricEnum):
        columns = ["machine_name", ]
        for metric in MetricEnum:
            columns.append(metric.value)
        self.metric_values = pd.DataFrame(columns=columns)
        self.metric_to_evaluate = metric_to_evaluate
        self.best_metric = -1
        self.best_machine = None
        if metric_to_evaluate == MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR:
            self.best_metric = 1000

    def add_metric_value(self, machine: MachineJson, metrics: Dict) -> None:
        """Add new metric and validate if is better that previous

        Args:
            machine (MachineJson): machine now
            metrics (Dict): metric for the machine
        """
        element = {"machine_name": str(machine)}
        element = element | metrics
        self.metric_values = self.metric_values.append(
            element, ignore_index=True)
        if self.metric_to_evaluate == MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR:
            self.validate_minus(
                value=metrics[self.metric_to_evaluate], machine=machine)

    def validate_minus(self, value: float, machine: MachineJson) -> None:
        """Validate if is better by minus values

        Args:
            value (float): new value
            machine (MachineJson): new machine
        """
        if self.best_metric > value:
            self.best_metric = value
            self.best_machine = machine


def combination_columns_from_data_frame(df: pd.DataFrame, min: int = 1):
    """Combine all columns with min of columns on combination

    Args:
        df (pd.DataFrame): data frame to get the columns 
        min (int, optional): minimal of column on group. Defaults to 1.

    Yields:
        _type_: return array of combination
    """
    columns = list(df.columns)
    for r in range(min, len(columns) + 1):
        for comb in itertools.combinations(columns, r):
            yield list(comb)

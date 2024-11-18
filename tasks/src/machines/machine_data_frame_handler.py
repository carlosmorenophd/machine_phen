"""Run machine"""

from sklearn.model_selection import train_test_split
import pandas as pd

from src.machines.machine_enums import (
    FileAccessRunnerProperties,
    DatasetProperties
)
from src.helpers.file_access import StorageFile


class DataFrameHandler():
    """Run all machines for Predicted
    """

    def __init__(
        self,
        file_access_runner: FileAccessRunnerProperties,
    ) -> None:
        self.file_access_runner = file_access_runner
        self.storage_file = StorageFile(
            file_name=file_access_runner.file_in,
            folder_file=file_access_runner.folder_path,
        )
        self.original_df = self.storage_file.get_csv_to_data_frame()
        x = self.get_data_frame_without_target()
        y = self.get_data_frame_only_target()
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=file_access_runner.test_size,
            random_state=file_access_runner.random_state
        )
        self._dataset = DatasetProperties(
            x=x,
            y=y,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

    @property
    def dataset(self) -> DatasetProperties:
        """Get dataset to work with it

        Returns:
            DatasetProperties: dataset
        """
        return self._dataset

    def change_column_from_data_frame(self, columns_to_keep: list):
        """Get some column of the dataset

        Args:
            columns_to_adding (list): list that columns to keep
        """
        x = self.get_data_frame_without_target()
        x = x[columns_to_keep]
        y = self.get_data_frame_only_target()
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.file_access_runner.test_size,
            random_state=self.file_access_runner.random_state
        )
        self._dataset = DatasetProperties(
            x=x,
            y=y,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

# Moving to other file
# Adding new function to save on file but adding result
    # def run_single_machine_single_file(
    #     self,
    #     machine_definition: MachineJson,
    # ) -> None:
    #     """Run single machine with single file

    #     Args:
    #         machine (MachineJson): machine to run 
    #     """
    #     print(f"Parameters machine- {machine_definition}")
    #     machine = build_machine(machine_definition=machine_definition)
    #     machine.build_machine()
    #     machine.training(self.dataset.x_train, self.dataset.y_train)
    #     error_metric = machine.test_machine(
    #         x_test=self.dataset.x_test,
    #         y_test=self.dataset.y_test,
    #     )
    #     print(f"Metric => {error_metric.metrics}")

    def get_data_frame_without_target(self) -> pd.DataFrame:
        """Get data frame without target columns

        Returns:
            pd.DataFrame: Data frame 
        """
        return self.original_df.drop(
            self.file_access_runner.target_feature,
            axis=1,
        )

    def get_data_frame_only_target(self) -> pd.DataFrame:
        """Get data frame only target column

        Returns:
            pd.DataFrame: Data frame 
        """
        return self.original_df[self.file_access_runner.target_feature]

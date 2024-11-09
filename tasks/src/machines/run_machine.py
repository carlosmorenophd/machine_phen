"""Run machine"""

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from numpy import ndarray
from pandas import DataFrame
from src.machines.predictions import RandomForestPrediction
from src.machines.enums import RandomForestJson, MachineNames
from src.helpers.file_access import FolderCache, get_file_to_data_frame



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
    df: DataFrame
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
        self.file_access = file_access_runner
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

    def run_single_machine_single_file(
        self,
        machine: RandomForestJson,

    ) -> None:
        """Run single machine with single file

        Args:
            machine (RandomForestJson): machine to run 
            file_in (str): file to get data
            folder_path (FolderCache, optional): 
                Folder to get the file. Defaults to FolderCache.UPLOAD.
        """
        if machine.name == MachineNames.RF:
            machine = RandomForestPrediction(
                n_estimators=machine.n_estimators,
                random_sate=machine.random_state,
                n_jobs=machine.n_jobs,
            )
            machine.build_machine()
            machine.training(self.dataset.x_train, self.dataset.y_train)
            machine.save_metric(
                x_test=self.dataset.x_test,
                y_test=self.dataset.y_test,
                base_file_name=self.file_access.file_name_only
            )

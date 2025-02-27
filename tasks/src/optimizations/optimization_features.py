"""Techniques for selection best features by models """

from src.files.file_machine import FileDataRegression
from src.helpers.file_access import FileData, StorageFile
from src.machines.machine_build import build_machine_regression
from src.machines.machine_enums import HyperParameterRebuild, convert_str_to_machine_name

# TODO: Implement FeatureForwardBackwardSelection
class FeatureForwardBackwardSelection:
    """Run the backward feature selection and whe it finish run the forward feature selection
    """

    def __init__(
        self,
        file_data: FileDataRegression,
        file_models: FileData,
    ) -> None:
        self._file_data = file_data
        self._file_models = file_models
        self._storage_file = StorageFile(
            file_name=self._file_data.file_in,
            folder_file=self._file_data.folder_path,
        )
        self.original_df = self._storage_file.get_csv_to_data_frame()
        self._storage_file_models = StorageFile(
            file_name=self._file_models.file_in,
            folder_file=self._file_models.folder_path,
        )
        self._original_df_models = self._storage_file_models.get_csv_to_data_frame()
        self._machine = None

    def run(self) -> None:
        """Run the backward feature selection and whe it finish run the forward feature selection
        """
        self.create_model()
        # self._backward_feature_selection()
        # self._forward_feature_selection()

    def _backward_feature_selection(self) -> None:
        """Run the backward feature selection
        """
        pass

    def _forward_feature_selection(self) -> None:
        """Run the forward feature selection
        """
        pass

    def create_model(self) -> None:
        """Create the model from file models
        """
        model_name = convert_str_to_machine_name(self._original_df_models.iloc[0, 0])
        parameters = []
        for index, column in enumerate(self._original_df_models.columns):
            if model_name.value in column:
                parameter = HyperParameterRebuild(
                    name=column.replace(f"_{model_name.value}", ''),
                    value=self._original_df_models.iloc[0, index]
                )
                parameters.append(parameter)
        self._machine = build_machine_regression(machine_name=model_name)
        self._machine.rebuild_machine(parameters=parameters)
        print(f"Machine : {self._machine}")

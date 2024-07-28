from machines.predictions import RF_Prediction
from preprocesses.preprocess import Preprocess
from preprocesses.enums import TransformEnum, TypeFileEnum, StandardScaleEnum
from metrics.error_metric import ErrorMetric
from metrics.enums import MetricEnum
from typing import List
import pandas as pd
from os import path


class Result_ML:
    def __init__(self, columns_name: List[str], file_to_save: str) -> None:
        self.columns_name = columns_name
        self.results = []
        self.adding_result = []
        self.file_to_save = file_to_save

    def adding_value(self, column_name: str, value):
        index = self.columns_name.index(column_name)
        self.adding_result.insert(index, value)

    def save_values(self):
        self.results.append(self.adding_result)
        self.adding_result = []

    def write_to_csv(self):
        df = pd.DataFrame(self.results, columns=self.columns_name)
        df.to_csv(self.file_to_save, index=False)


class Do_Run_ML:
    def __init__(
        self,
        file_name_save: str,
        is_debug: bool = False,
    ) -> None:
        self.is_debug = is_debug
        self.files_dataset = {}
        self.machines = []
        self.results = Result_ML(
            columns_name=[
                "file_key",
                "file_name",
                "ml",
                "time_training",
                "RMSE",
                "R2",
                "MAPE",
            ],
            file_to_save=file_name_save,
        )

    def adding_dataset(
        self,
        key: str,
        names: List[str],
        type: TypeFileEnum = TypeFileEnum.CSV,
        transform: TransformEnum = TransformEnum.MEAN,
        standard_scale: StandardScaleEnum = StandardScaleEnum.BASIC,
        is_search_best_pca_component: bool = False,
    ):
        if not key in self.files_dataset:
            self.files_dataset[key] = {
                "type": type,
                "datasets": names,
                "transform": transform,
                "standard_scale": standard_scale,
                "is_search_best_pca_component": is_search_best_pca_component,
            }
        else:
            raise NameError("Duplicate name os key on files dataset")

    def adding_machine_dataset(self, machine, key_dataset: str):
        self.machines.append(
            {"machine": machine, "key_dataset": key_dataset, "all": False}
        )

    def run(self):
        for key_file in self.files_dataset:
            for file in self.files_dataset[key_file]["datasets"]:
                preprocessing = Preprocess(
                    file_name=file,
                    type_file=self.files_dataset[key_file]["type"],
                )
                preprocessing.read_file(
                    transform=self.files_dataset[key_file]["transform"],
                    standard_scale=self.files_dataset[key_file]["standard_scale"],
                    is_search_best_pca_component=self.files_dataset[key_file][
                        "is_search_best_pca_component"
                    ],
                )
                machine_to_run = filter(
                    lambda machine: machine["all"] == True
                    or machine["key_dataset"] == key_file,
                    self.machines,
                )
                for machine in machine_to_run:
                    self.results.adding_value(column_name="file_key", value=key_file)
                    self.results.adding_value(
                        column_name="file_name", value=path.basename(file)
                    )
                    preprocessing.build_train_and_test()
                    x_train, y_train = preprocessing.get_train()
                    x_test, y_test = preprocessing.get_test()
                    ml = machine["machine"]
                    ml.training(x_train=x_train, y_train=y_train)
                    self.results.adding_value(
                        column_name="time_training", value=ml.time_training
                    )
                    self.results.adding_value(column_name="ml", value=str(ml))
                    y_predicted = ml.prediction(x_test=x_test)
                    metric = ErrorMetric(
                        y_predicted=y_predicted,
                        y_test=y_test,
                        x_test=x_test,
                    )
                    metric.calculate_metric_prediction()
                    self.results.adding_value(
                        column_name="RMSE",
                        value=metric.get_metric(
                            metric=MetricEnum.ROOT_MEAN_SQUARED_ERROR
                        ),
                    )
                    self.results.adding_value(
                        column_name="R2",
                        value=metric.get_metric(metric=MetricEnum.R2_SCORE),
                    )
                    self.results.adding_value(
                        column_name="MAPE",
                        value=metric.get_metric(
                            metric=MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR
                        ),
                    )
                    self.results.save_values()
        self.results.write_to_csv()

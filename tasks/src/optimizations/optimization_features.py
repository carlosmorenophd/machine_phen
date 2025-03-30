"""Techniques for selection best features by models """

from typing import List, Dict
import copy

from src.optimizations.optimization_enum import (
    FileProcessForwardBackward,
    ActionProcedure,
)
# from src.metrics.metric_enums import MetricEnum
from src.machines.machine_build import build_machine_regression
from src.files.file_machine import DatasetOptimizationData
from src.machines.machine import MachineRegression


class FeatureForwardBackwardSelection:
    """Run the backward feature selection and whe it finish
        run the forward feature selection
    """

    def __init__(
        self,
        file_process: FileProcessForwardBackward,
    ) -> None:
        self._procedure = file_process.procedure
        self._file_machine = file_process.file_machine
        self._result_dictionary = []

    def run(self) -> None:
        """Run the backward feature selection and whe it finish run
        the forward feature selection
        """
        if self._procedure.action == ActionProcedure.BACKWARD:
            self._backward_feature_selection()

    def _backward_feature_selection(
            self,
    ) -> None:
        """Run the backward feature selection
        """
        features_chromosome = self._file_machine.cast_features_to_chromosome(
            features=self._procedure.features
        )
        for model in self._procedure.models:
            machine = build_machine_regression(
                machine_name=model.machine_name
            )
            if len(model.rebuild_hyper_parameters) > 0:
                machine.rebuild_hyper_parameters(
                    rebuild_parameters=model.rebuild_hyper_parameters
                )
            self._run_model(
                features_chromosome=features_chromosome,
                machine=machine,
            )

    def _run_model(
            self,
            features_chromosome: List[bool],
            machine: MachineRegression,
    ) -> None:
        """Run one model with all combination features

        Args:
            features (List[bool]): List of features
            machine (MachineRegression): machine to run
        """
        data_set = self._file_machine.get_dataset(
            columns_to_keep=features_chromosome,
        )
        machine.build_machine_default()
        machine.training(
            x_train=data_set.x_train,
            y_train=data_set.y_train,
        )
        metric = machine.test(
            x_test=data_set.x_test,
            y_test=data_set.y_test,
        )
        best_metric = metric.get_single_metric(
            metric=self._procedure.metric,
        )
        best_features_chromosome = copy.deepcopy(features_chromosome)
        self._result_dictionary.append(
            self._get_id_to_save(
                machine=machine,
                features_chromosome=best_features_chromosome,
                metrics=metric.get_all_metric(),
            )
        )

        for index, value in enumerate(features_chromosome):
            if value is True:
                machine.build_machine_default()
                best_features_chromosome[index] = False
                data_set = self._file_machine.get_dataset(
                    columns_to_keep=best_features_chromosome,
                )
                machine.build_machine_default()
                machine.training(
                    x_train=data_set.x_train,
                    y_train=data_set.y_train,
                )
                metric = machine.test(
                    x_test=data_set.x_test,
                    y_test=data_set.y_test,
                )
                self._result_dictionary.append(
                    self._get_id_to_save(
                        machine=machine,
                        features_chromosome=best_features_chromosome,
                        metrics=metric.get_all_metric(),
                    )
                )
                if best_metric < metric:
                    best_metric = metric
                else:
                    best_features_chromosome[index] = True

    def _get_id_to_save(
            self,
            machine: MachineRegression,
            features_chromosome: List[bool],
            metrics: Dict,
    ) -> Dict:
        dictionary_id = {
            "machine_name": machine.machine_name,
            "time_training": machine.time_training,
        }
        for rebuild_hyper_parameter in machine.rebuild_hyper_parameters:
            key_id = f"{rebuild_hyper_parameter.name}_{machine.machine_name}"
            dictionary_id[key_id] =\
                rebuild_hyper_parameter.value
        for chromosome, feature in zip(
            features_chromosome,
            self._file_machine.columns_name_with_out_target
        ):
            dictionary_id[f"feature_{feature}"] = chromosome
        for key, value in metrics.items():
            dictionary_id[f"feature_{key}"] = value
        return dictionary_id

    def _forward_feature_selection(self) -> None:
        """Run the forward feature selection
        """
        pass

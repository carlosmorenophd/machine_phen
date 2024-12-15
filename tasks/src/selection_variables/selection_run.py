"""Runner to select variable variable"""


from src.machines.machine_enums import MachineJson, FileAccessRunnerProperties
from src.machines.machine_data_frame_handler import DataFrameHandler
from src.machines.regression_run import machine_build_regression
from src.metrics.metric_enums import MetricEnum
from src.selection_variables.selection_data_frame_handler import SelectionBestMetric
from src.selection_variables.force_brute import combination_columns_from_data_frame
from src.selection_variables.genetic.genetic_enum import GeneticParameter, MachineMainRegression
from src.selection_variables.genetic.genetic_models import GeneticAlgorithm


def selection_force_brute_run(
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
    best_metric = SelectionBestMetric(metric_to_evaluate=metric)
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
        best_metric.add_metric_value(
            machine_definition=machine_definition,
            columns=combination,
            metrics=error_metric.metrics,
        )
    df_metric = best_metric.metric_values
    data_frame.storage_file.save_data_frame_to_csv(
        data_frame=df_metric, prefix="metric_force_selection",
    )


def selection_genetic_algorithm_run(
    file_access_runner: FileAccessRunnerProperties,
    machine_definition: MachineJson,
    genetic_parameters: GeneticParameter,
    metric: MetricEnum = MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
) -> None:
    """Search best combination of variables on dataset and 

    Args:
        file_access_runner (FileAccessRunnerProperties): file access
        machine_definition (MachineJson): machine
        metric (MetricEnum, optional): metric to evaluate. 
            Defaults to MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR.
    """
    machine_main = MachineMainRegression(
        data_frame=DataFrameHandler(file_access_runner=file_access_runner),
        best_metric=SelectionBestMetric(metric_to_evaluate=metric),
        machine_definition=machine_definition,
        machine=machine_build_regression(machine_definition=machine_definition)
    )

    genetic_parameters.chromosome_length = len(
        machine_main.data_frame.get_data_frame_without_target().columns)
    genetic = GeneticAlgorithm(
        machine_main=machine_main,
        parameter=genetic_parameters,
    )
    genetic.run()

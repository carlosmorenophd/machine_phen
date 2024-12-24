"""Run task like machine and populate result"""
import gc
import json


from celery import Celery
from src.helpers.key_env import REDIS_BROKEN, FolderCache
from src.machines.machine_enums import build_machine_definition
from src.machines.machine_data_frame_handler import FileAccessRunnerProperties
from src.selection_variables.selection_run import (
    selection_force_brute_run,
    selection_genetic_algorithm_run,
)
from src.selection_variables.genetic.genetic_enum import convert_str_genetic_parameters


print(REDIS_BROKEN)

app = Celery('phen_machine', broker=REDIS_BROKEN)


@app.task(name="result_single-machine-single-file")
def result_single_machine_file_single(files_in: str, target_column: str, machine_str: str) -> None:
    """Run on file a list of files split """
    print(
        f" Inputs: file - {files_in}, column - {
            target_column}, machine - {machine_str}"
    )
    machine_definition = build_machine_definition(
        machine_json=json.loads(machine_str))

    file_access_runner = FileAccessRunnerProperties(
        file_in=files_in,
        target_feature=target_column,
        folder_path=FolderCache.UPLOAD,
    )
    # data_frame = DataFrameHandler(file_access_runner=file_access_runner)
    # machine.(
    #     machine_definition=machine_definition)
    # del machine_definition
    gc.collect()


@app.task(name="regression_forward_force_single_machine_single_file")
def task_regression_forward_force_single_machine_single_file(
    files_in: str,
    target_column: str,
    machine_str: str
) -> None:
    """Run a forward selection in force brut to get the best variables to predict

    Args:
        files_in (str): file to get data
        target_column (str): column target
        machine_str (str): machine description on str
    """
    print(
        f" Inputs: file - {files_in}, column - {
            target_column}, machine - {machine_str}"
    )
    selection_force_brute_run(
        machine_definition=build_machine_definition(
            machine_json=json.loads(machine_str)
        ),
        file_access_runner=FileAccessRunnerProperties(
            file_in=files_in,
            target_feature=target_column,
            folder_path=FolderCache.UPLOAD,
        ),
    )
    gc.collect()


@app.task(name="regression_genetic_single_machine_single_file")
def task_regression_genetic_single_machine_single_file(
    files_in: str,
    target_column: str,
    machine_str: str,
    genetic_parameters_str: str,
) -> None:
    """Run a forward selection in force brut to get the best variables to predict

    Args:
        files_in (str): file to get data
        target_column (str): column target
        machine_str (str): machine description on str
    """
    print(
        f" Inputs: file - {files_in}, column - {
            target_column}, machine - {machine_str}"
    )
    selection_genetic_algorithm_run(
        machine_definition=build_machine_definition(
            machine_json=json.loads(machine_str)
        ),
        file_access_runner=FileAccessRunnerProperties(
            file_in=files_in,
            target_feature=target_column,
            folder_path=FolderCache.UPLOAD,
        ),
        genetic_parameters=convert_str_genetic_parameters(
            genetic_parameters_str=genetic_parameters_str),
    )
    gc.collect()


@app.task(name="version")
def version() -> str:
    """Version of system

    Returns:
        str: version number
    """
    return "24.12.141"

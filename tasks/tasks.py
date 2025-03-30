"""Run task like machine and populate result"""
import gc


from celery import Celery
from src.helpers.key_env import REDIS_BROKEN, FolderCache
from src.files.file_machine import FileDataRegression
from src.optimizations.optimization import optimization_run_from_task, forward_backward_features


app = Celery('phen_machine', broker=REDIS_BROKEN, queue='machine')


@app.task(name="regression_genetic")
def task_regression_genetic(
    files_in: str,
    target_column: str,
) -> None:
    """Search the best machine for regression on one file

    Args:
        files_in (str): file to get data
        target_column (str): column target on file
        genetic_parameters_str (str): basic parameters for genetic algorithm
    """
    print(f" Inputs: file - {files_in}, column - {target_column}")
    optimization_run_from_task(
        file_data=FileDataRegression(
            file_in=files_in,
            target_feature=target_column,
            folder_path=FolderCache.UPLOAD,
        ),
    )
    gc.collect()


@app.task(name="feature_selection")
def task_feature_selection(
    file_data: str,
    file_json_definition: str,
) -> None:
    """Search the best features selection with backward adn forward action 

    Args:
        files_in (str): file to get data
        target_column (str): column target on file
        genetic_parameters_str (str): basic parameters for genetic algorithm
    """
    log_print = f"{log_print} file data - {file_data}"
    log_print = f"{log_print} file json definition - {file_json_definition}"
    print(log_print)
    forward_backward_features(
        folder_main,
        file_data,
        file_json_definition,
    )
    gc.collect()


@app.task(name="version")
def version() -> str:
    """Version of system

    Returns:
        str: version number
    """
    return "25.03.28"

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


@app.task(name="back_forward")
def task_back_forward_feature(
    files_in: str,
    target_column: str,
    file_models: str,
) -> None:
    """Search the best features selection with backward adn forward action 

    Args:
        files_in (str): file to get data
        target_column (str): column target on file
        genetic_parameters_str (str): basic parameters for genetic algorithm
    """
    print(f" Inputs: file - {files_in}, column - {
          target_column}, file models - {file_models}")
    forward_backward_features(
        file_data=FileDataRegression(
            file_in=files_in,
            target_feature=target_column,
            folder_path=FolderCache.UPLOAD,
        ),
        file_models=FileDataRegression(
            file_in=file_models,
            target_feature=target_column,
            folder_path=FolderCache.UPLOAD,
        ),
    )
    gc.collect()


@app.task(name="version")
def version() -> str:
    """Version of system

    Returns:
        str: version number
    """
    return "25.01.171"

"""Run task like machine and populate result"""
import gc
import json


from celery import Celery
from src.helpers.key_env import REDIS_BROKEN, FolderCache
from src.helpers.file_access import get_name_file_without_extension
from src.machines.enums import (
    RandomForestJson,
    MachineNames,
    ExtremeGradientBoost,
    BayesianPredictionDefinition,
    LassoPredictionDataClass,
    SupportVectorRegressionPredictionDataClass,
    cast_kernel_svr,
)
from src.machines.run_machine import MachineRunner, FileAccessRunnerProperties


print(REDIS_BROKEN)

app = Celery('phen_machine', broker=REDIS_BROKEN)


@app.task(name="result_single-machine-single-file")
def result_single_machine_file_single(files_in: str, target_column: str, machine_str: str) -> None:
    """Run on file a list of files split """
    print(
        f" Inputs: file - {files_in}, column - {
            target_column}, machine - {machine_str}"
    )
    machine_json = json.loads(machine_str)
    machine_definition = None
    if "name" not in machine_json:
        raise NotImplementedError("Not have a name of machine")
    if machine_json["name"] == MachineNames.RF.value:
        machine_definition = RandomForestJson()
        if "parameters" in machine_json:
            parameters = machine_json["parameters"]
            if "n_estimators" in parameters:
                machine_definition.n_estimators = parameters["n_estimators"]
            if "random_state" in parameters:
                machine_definition.random_state = parameters["random_state"]
            if "n_jobs" in parameters:
                machine_definition.n_jobs = parameters["n_jobs"]
    elif machine_json["name"] == MachineNames.XGB.value:
        machine_definition = ExtremeGradientBoost()
        if "parameters" in machine_json:
            parameters = machine_json["parameters"]
            if "n_estimators" in parameters:
                machine_definition.n_estimators = parameters["n_estimators"]
            if "max_leaves" in parameters:
                machine_definition.max_leaves = parameters["max_leaves"]
            if "max_depth" in parameters:
                machine_definition.max_depth = parameters["max_depth"]
    elif machine_json["name"] == MachineNames.BAP.value:
        machine_definition = BayesianPredictionDefinition()
    elif machine_json["name"] == MachineNames.LAP.value:
        machine_definition = LassoPredictionDataClass()
        if "parameters" in machine_json:
            parameters = machine_json["parameters"]
            if "alpha" in parameters:
                machine_definition.alpha = parameters["alpha"]
    elif machine_json["name"] == MachineNames.SVRP.value:
        machine_definition = SupportVectorRegressionPredictionDataClass()
        if "parameters" in machine_json:
            parameters = machine_json["parameters"]
            if "kernel" in parameters:
                machine_definition.kernel = cast_kernel_svr(input_kernel = parameters["kernel"])
            if "c" in parameters:
                machine_definition.c = parameters["c"]
            if "epsilon" in parameters:
                machine_definition.c = parameters["epsilon"]

    name_file_without_extension = get_name_file_without_extension(
        file_name=files_in, folder=FolderCache.UPLOAD
    )
    file_access_runner = FileAccessRunnerProperties(
        file_in=files_in,
        target_feature=target_column,
        folder_path=FolderCache.UPLOAD,
        file_name_only=name_file_without_extension
    )
    machine = MachineRunner(file_access_runner=file_access_runner)
    machine.run_single_machine_single_file(
        machine_definition=machine_definition)
    del machine_definition
    gc.collect()


@app.task(name="version")
def version() -> str:
    """Version of system

    Returns:
        str: version number
    """
    return "24.11.09"

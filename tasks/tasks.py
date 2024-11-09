"""Run task like machine and populate result"""
import gc
import json


from celery import Celery
from src.helpers.key_env import REDIS_BROKEN, FolderCache
from src.helpers.file_access import get_name_file_without_extension
from src.machines.enums import RandomForestJson, MachineNames
from src.machines.run_machine import MachineRunner, FileAccessRunnerProperties


print(REDIS_BROKEN)

app = Celery('phen_transform', broker=REDIS_BROKEN)


@app.task(name="result_single-machine")
def machine_single_file_single(files_in: str, target_column: str, machine_str: str) -> None:
    """Run on file a list of files split """
    machine_json = json.loads(machine_str)
    machine_build = None
    if "name" not in machine_json:
        raise NotImplementedError("Not have a name of machine")
    if machine_json["name"] == MachineNames.RF.value:
        machine_build = RandomForestJson(name=MachineNames.RF)
        if "parameters" in machine_json:
            parameters = machine_json["parameters"]
            if "n_estimators" in parameters:
                machine_build.n_estimators = parameters["n_estimators"]
            if "random_state" in parameters:
                machine_build.random_state = parameters["random_state"]
            if "n_jobs" in parameters:
                machine_build.n_jobs = parameters["n_jobs"]
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
        machine.run_single_machine_single_file(machine=machine_build)
    del machine_build
    gc.collect()

@app.task(name="version")
def version() -> str:
    return "24.11"

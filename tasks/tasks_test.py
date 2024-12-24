"""To run test from files"""
import sys
import json

from celery.app import Celery

from src.helpers.key_env import REDIS_BROKEN, FolderCache
from src.selection_variables.selection_run import selection_genetic_algorithm_run
from src.machines.machine_enums import build_machine_definition
from src.machines.machine_data_frame_handler import FileAccessRunnerProperties
from src.selection_variables.genetic.genetic_enum import convert_str_genetic_parameters

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f"Run with props action - {sys.argv[1]}")
        app = Celery('phen_transform', broker_url=REDIS_BROKEN)
        action = sys.argv[1]
        if action == "version":
            print("Run - version")
            app.send_task(
                name="version",
            )
        elif action == "result_single-machine-single-file":
            print("Run - result_single-machine-single-file")
            print(f"file -> {
                sys.argv[2]
            }, target -> {
                sys.argv[3]
            }, machine - {
                sys.argv[4]
            }")
            app.send_task(
                name="result_single-machine-single-file",
                args=(
                    sys.argv[2],
                    sys.argv[3],
                    sys.argv[4],
                )
            )
        elif action == "regression_forward_force_single_machine_single_file":
            print("Run - regression_forward_force_single_machine_single_file")
            print(f"file -> {
                sys.argv[2]
            }, target -> {
                sys.argv[3]
            }, machine - {
                sys.argv[4]
            }")
            app.send_task(
                name="regression_forward_force_single_machine_single_file",
                args=(
                    sys.argv[2],
                    sys.argv[3],
                    sys.argv[4],
                )
            )
        elif action == "regression_genetic_single_machine_single_file":
            print("Run - regression_genetic_single_machine_single_file")
            print(f"file -> {
                sys.argv[2]
            }, target -> {
                sys.argv[3]
            }, machine - {
                sys.argv[4]
            }")
            app.send_task(
                name="regression_genetic_single_machine_single_file",
                args=(
                    sys.argv[2],
                    sys.argv[3],
                    sys.argv[4],
                    sys.argv[5],
                )
            )
    else:
        # python tasks_test.py regression_genetic_single_machine_single_file    
        selection_genetic_algorithm_run(
            machine_definition=build_machine_definition(
                machine_json=json.loads('{"name": "random_forest_regression"}')
            ),
            file_access_runner=FileAccessRunnerProperties(
                file_in='lrace_all_clean_fill_normalize.csv',
                target_feature='GrainYield',
                folder_path=FolderCache.UPLOAD,
            ),
            genetic_parameters=convert_str_genetic_parameters(
                genetic_parameters_str='{"num_generations":5}'),
        )

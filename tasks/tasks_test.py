"""To run test from files"""
import sys

from celery.app import Celery

from src.helpers.key_env import REDIS_BROKEN
from src.optimizations.optimization import optimization_run_from_task
from src.optimizations.optimization_enum import convert_str_to_search_mode
from src.files.file_machine import FileData, FolderCache

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f"Run with props action - {sys.argv[1]}")
        app = Celery('phen_transform',
                     broker_url=REDIS_BROKEN, queue='machine')
        action = sys.argv[1]
        if action == "version":
            print("Run - version")
            app.send_task(
                name="version",
                args=(),
                queue='machine',
            )
        elif action == "regression_genetic":
            print("Run - regression_genetic")
            app.send_task(
                name="regression_genetic",
                args=(
                    sys.argv[2],
                    sys.argv[3],
                    sys.argv[4],
                ),
                queue='machine',
            )
# python tasks_test.py regression_genetic 3.14_lrace_geo_w_f_n.csv Rendimiento basic_search
    else:
        print("No action to run")
        optimization_run_from_task(
            file_date=FileData(
                file_in="3.14_lrace_geo_w_f_n.csv",
                target_feature="Rendimiento",
                folder_path=FolderCache.UPLOAD,
            ),
            search_mode=convert_str_to_search_mode(
                search_mode_str="quick_exploration",
            ),
        )

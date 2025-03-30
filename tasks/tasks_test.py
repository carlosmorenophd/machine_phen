"""To run test from files"""
import sys

from celery.app import Celery

from src.helpers.key_env import REDIS_BROKEN
# from src.optimizations.optimization import forward_backward_features
from src.files.file_machine import FileDataRegression
from src.helpers.key_env import FolderCache
# from src.helpers.file_access import FileData
from src.optimizations.optimization import optimization_run_from_task
from src.results.result_genetic import create_graph_model_index

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
        if action == "regression_genetic":
            print("Run - regression_genetic")
            app.send_task(
                name="regression_genetic",
                args=(
                    sys.argv[2],
                    sys.argv[3],
                ),
                queue='machine',
            )
        if action == "feature_selection":
            print("Run - feature_selection")
            app.send_task(
                name="feature_selection",
                args=(
                    sys.argv[2],
                    sys.argv[3],
                ),
                queue='machine',
            )
    else:
        print("No action to run")
        optimization_run_from_task(
            file_data=FileDataRegression(
                file_in="3.1.LRACE_FA_W_FMN.csv",
                target_feature="RendimientoKG/HA",
                folder_path=FolderCache.UPLOAD,
            ),
        )
        print("Finish")

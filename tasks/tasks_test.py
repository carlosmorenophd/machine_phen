"""To run test from files"""
import sys

from celery.app import Celery
from src.helpers.key_env import REDIS_BROKEN

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f"Run with props action - {sys.argv[1]}")
        app = Celery('phen_transform', broker_url=REDIS_BROKEN)
        action = sys.argv[1]
        if action == "result_single-machine-single-file":
            print("Run - result_single-machine-single-file")
            # python tasks_test.py result_single-machine-single-file pca_lrace_trueba_fill_clean.csv Yield '{"name": "random_forest"}' 
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

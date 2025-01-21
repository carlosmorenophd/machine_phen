"""To run test from files"""
import sys

from celery.app import Celery

from src.helpers.key_env import REDIS_BROKEN

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

    else:
        print("No action to run")

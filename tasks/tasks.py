import gc

from celery import Celery
from src.helpers.file_access import REDIS_BROKEN


print(REDIS_BROKEN)

app = Celery('phen_transform', broker=REDIS_BROKEN)


@app.task(name="result_single-machine")
def result_single_machine(files_str: str, machines_str: str) -> None:
    files = files_str.split(",")
    machines = machines_str.split(",")
    

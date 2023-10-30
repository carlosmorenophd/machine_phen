import numpy as np
import pandas as pd
from typing import Tuple
from numpy import ndarray

def get_from_csv(file_name: str) -> Tuple[ndarray, ndarray]:
    dataset = pd.read_csv(file_name)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y


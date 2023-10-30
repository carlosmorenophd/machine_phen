import numpy as np
import pandas as pd
from typing import Tuple
from numpy import ndarray
from sklearn.decomposition import PCA

def get_file_data(file_name: str) -> Tuple[ndarray, ndarray]:
    dataset = pd.read_csv(file_name)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y

def reducer_by_pca(X, n_components: int = 2):
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    return X
from sklearn.decomposition import PCA
from numpy import abs, ndarray
from typing import Tuple


class PCA_Preprocessing():
    def __init__(self, data: ndarray, is_debug: bool = False) -> None:
        self.data = data
        self.is_debug = is_debug

    def get_transform(self, n_components: int = 2) -> ndarray:
        pca = PCA(n_components=n_components)
        return pca.fit_transform(self.data)

    def evaluate_pca(self, n_components: int = 2) -> Tuple[ndarray, ndarray] : 
        pca = PCA(n_components=n_components)
        pca.fit(self.data)
        variance_ratio = pca.explained_variance_ratio_
        if self.is_debug:
            print("Variance of PCA => ", variance_ratio)
        weights = pca.components_[0]
        most_important_columns = abs(weights).argsort()[::-1]
        return most_important_columns, variance_ratio

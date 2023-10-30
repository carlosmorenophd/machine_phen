from sklearn.decomposition import PCA
import numpy as np

def get_by_pca(X, n_components: int = 2):
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    return X

def get_by_pca_best_columns(X, n_components: int = 2):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    variance_ratio = pca.explained_variance_ratio_
    weights = pca.components_[0]
    most_important_columns = np.abs(weights).argsort()[::-1]
    return most_important_columns, variance_ratio
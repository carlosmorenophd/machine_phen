from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from model import ResultSVM
from numpy import ndarray
from sklearn.decomposition import PCA
from typing import Tuple


def get_file_data(file_name: str) -> Tuple[ndarray, ndarray]:
    dataset = pd.read_csv(file_name)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y


def calculate_svm_simple(file_name: str, debug: bool = False):
    X, y = get_file_data(file_name=file_name)
    return calculate_svm(X=X, y=y, debug=debug)


def calculate_svm_principal_component_analysis(file_name: str, debug: bool = False, n_components: int = 2):
    X, y = get_file_data(file_name=file_name)
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    return calculate_svm(X=X, y=y, debug=debug)


def calculate_svm(X: ndarray, y: ndarray, debug: bool = False) -> ResultSVM:
    """_summary_ Calculate SVM return matrix confusion and accuracy
    """

    if debug:
        print("Variable X =>", X)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X)
    X = imputer.transform(X)

    if debug:
        print("Transform variable X =>", X)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    if debug:
        print("Training data => ", X_train)
        print("Training y => ", y_train)
        print("Set to test => ", X_test)
        print("Test y => ", y_test)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    if debug:
        print("Scaling training set =>", X_train)
        print("Scaling test set =>", X_test)

    # Training the Kernel SVM model on the Training set
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    if debug:
        print("Result of testing => ")
        print(np.concatenate((y_pred.reshape(len(y_pred), 1),
                              y_test.reshape(len(y_test), 1)), 1))

    result = ResultSVM()
    result.calculate_all_basic(y_test=y_test, y_pred=y_pred)
    return result

"""Preprocessing data to machines"""
from typing import Tuple
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from preprocesses.enums import TypeFileEnum, TransformEnum, StandardScaleEnum
from preprocesses.pca_preprocess import PCA_Preprocess


class Preprocess:
    """Main class for preprocessing"""

    def __init__(
        self, file_name: str, type_file: TypeFileEnum, is_debug: bool = False
    ) -> None:
        self.file_name = file_name
        self.type_file = type_file
        self.is_debug = is_debug
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_transform = None
        self.transform = None
        self.dataset = None
        self.x = None
        self.y = None
        self.imputer = None
        self.pca = None

    def read_file(
        self,
        transform: TransformEnum = TransformEnum.PASS,
        standard_scale: StandardScaleEnum = StandardScaleEnum.PASS,
        is_search_best_pca_component: bool = False,
    ) -> None:
        """Read file"""
        self.transform = transform
        if self.type_file == TypeFileEnum.CSV:
            self.dataset = pd.read_csv(self.file_name)
            if self.is_debug:
                print(
                    "Number of element with NAN values => ",
                    self.dataset.isnull().sum().sum(),
                )
            self.x = self.dataset.iloc[:, :-1].values
            self.y = self.dataset.iloc[:, -1].values
            if self.is_debug:
                print("Number of element  => ", self.x.shape, self.y.shape)
        if transform == TransformEnum.MEAN:
            self.imputer = SimpleImputer(strategy=transform.value)
            self.x_transform = self.imputer.fit_transform(self.x)
        elif transform == TransformEnum.PCA:
            self.imputer = SimpleImputer(strategy=TransformEnum.MEAN.value)
            self.x_transform = self.imputer.fit_transform(self.x)
            headers = pd.read_csv(self.file_name, header=None).iloc[0]
            self.pca = PCA_Preprocess(
                data=self.x_transform,
                target=self.y,
                feature_names=headers,
                is_debug=self.is_debug,
            )
            self.x_transform = self.pca.get_transform(
                is_search_best_component=is_search_best_pca_component
            )
        else:
            self.x_transform = self.x

        if standard_scale == StandardScaleEnum.BASIC:
            self.scaler = StandardScaler().fit(self.x_transform)
            self.x_transform = self.scaler.transform(self.x_transform)
        else:
            self.x_transform = self.x_transform

    def set_data(
        self,
        x,
        y,
    ) -> None:
        """Function to set data"""
        self.x = x
        self.y = y
        self.x_transform = x

    def build_train_and_test(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> None:
        """Create train and test"""
        self.test_size = test_size
        self.random_state = random_state
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_transform, self.y, test_size=test_size, random_state=random_state
        )

    def get_parameter_train_test(self) -> Tuple:
        """Get parameter for train and test"""
        return self.test_size, self.random_state

    def get_pca(self, threshold: float = 0.95):
        """Get from PCA"""
        if self.transform == TransformEnum.PCA:
            self.pca.evaluate_pca(n_components=None, threshold=threshold)
            return self.pca

    def get_train(self):
        """Get train"""
        if self.x_train is None:
            raise ModuleNotFoundError(
                "Sorry, you need to set the method build_train_and_test before call this method"
            )
        return self.x_train, self.y_train

    def get_test(self):
        """Get test"""
        if self.x_train is None:
            raise ModuleNotFoundError(
                "Sorry, you need to set the method set_train_and_test before call this method"
            )
        return self.x_test, self.y_test

    def get_name_of_column_by_index(self, list_index: list) -> list:
        """Get name of columns by index"""
        column_header = self.dataset.columns.values
        result = []
        for index in list_index:
            result.append(column_header[index])
        return result

    def get_name_of_column(self) -> list:
        """Get list of columns """
        return self.dataset.columns.values

    def get_name_features(self) -> list:
        """Get name of features"""
        return self.dataset.columns.values[:-1]

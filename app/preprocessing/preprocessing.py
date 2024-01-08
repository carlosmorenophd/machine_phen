import pandas as pd
from numpy import ndarray
from sklearn.impute import SimpleImputer
from preprocessing.enums import TypeFileEnum, TransformEnum
from sklearn.model_selection import train_test_split
from typing import Tuple




class Preprocessing():
    def __init__(self, file_name: str, type_file: TypeFileEnum, is_debug: bool = False) -> None:
        self.file_name = file_name
        self.type_file = type_file
        self.is_debug = is_debug
        self.x_train = None
        self.x_test = None
        self.y_train = None 
        self.y_test = None

    def read_file(self, transform:  str = "") -> None:
        self.transform = transform
        if self.type_file == TypeFileEnum.CSV:
            self.dataset = pd.read_csv(self.file_name)
            if self.is_debug:
                print("Number of element with NAN values => ",
                      self.dataset.isnull().sum().sum())
            self.x = self.dataset.iloc[:, :-1].values
            self.y = self.dataset.iloc[:, -1].values
            if self.is_debug:
                print("Number of element  => ",
                      self.x.shape, self.y.shape)
        if transform != TransformEnum.PASS:
            imputer = SimpleImputer(strategy=transform.value)
            self.x_transform = imputer.fit_transform(self.x)

    def get_x(self) -> ndarray:
        return self.x
    
    def get_y(self) -> ndarray:
        return self.y
    
    def get_x_transform(self) -> ndarray :
        return self.x_transform
    
    def set_train_and_test(self, test_size: float =0.2, random_state: int=42) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=random_state)

    def get_parameter_train_test(self) -> Tuple:
        return self.test_size, self.random_state
    
    def get_train(self):
        if self.x_train == None:
            raise Exception("Sorry, you need to set the method set_train_and_test before call this method")
        else:
            return self.x_train, self.y_train
        
    def get_test(self):
        if self.x_train == None:
            raise Exception("Sorry, you need to set the method set_train_and_test before call this method")
        else:
            return self.x_test, self.y_test

    def get_name_of_column_by_index(self, list_index):
        column_header = self.dataset.columns.values
        result = []
        for index in list_index:
            result.append(column_header[index])
        return result
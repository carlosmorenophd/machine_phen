from enum import Enum


class SvrKernelEnum(Enum):
    RBF = 'rbf'
    LINEAR = 'linear'
    POLY = 'poly'
    SIGMOID = 'sigmoid'
    PRE_COMPUTER = 'precomputed'

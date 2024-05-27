from enum import Enum


class TransformEnum(Enum):
    PASS = ''
    MEAN = 'mean'
    PCA = 'pca'


class StandardScaleEnum(Enum):
    PASS = ''
    BASIC = 'basic'

class TypeFileEnum(Enum):
    CSV = 0

"""DEfinition of all actions"""
from enum import Enum


class TransformEnum(Enum):
    """Actions for transformation"""
    PASS = ''
    MEAN = 'mean'
    PCA = 'pca'


class StandardScaleEnum(Enum):
    """Action for basic standard transform"""
    PASS = ''
    BASIC = 'basic'


class TypeFileEnum(Enum):
    """Support files"""
    CSV = 0

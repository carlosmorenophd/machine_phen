from enum import Enum


class TransformEnum(Enum):
    PASS = ''
    MEAN = 'mean'


class StandardScaleEnum(Enum):
    PASS = ''
    BASIC = 'basic'

class TypeFileEnum(Enum):
    CSV = 0

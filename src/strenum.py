from enum import Enum


class StrEnum(str, Enum):
    """Enum and string subclass to ease display, saving and comparisons"""

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def strenum(name, values):
    return StrEnum(name, [(v, v) for v in values.split(' ')])

from abc import ABC, abstractmethod

__all__ = [
    '_Report'
]


class _Report(ABC):
    @abstractmethod
    def __init__(self):
        pass

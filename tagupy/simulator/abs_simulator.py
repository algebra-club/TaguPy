from abc import ABC, abstractmethod

__all__ = [
    "_Simulator"
]


class _Simulator(ABC):
    @abstractmethod
    def __init__(self):
        pass

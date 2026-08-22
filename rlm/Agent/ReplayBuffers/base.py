
from abc import ABC, abstractmethod


class RBuffer(ABC):
    def __init__(self, capacity, removals):
        pass

    @abstractmethod
    def add(self, tuple):
        pass

    @abstractmethod
    def sample(self, sample_size:int):
        pass
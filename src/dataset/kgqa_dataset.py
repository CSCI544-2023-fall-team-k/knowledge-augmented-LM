from abc import ABC, abstractmethod
from typing import List

from .kgqa_data import KGQAData

class KGQADataSet(ABC):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.data: List[KGQAData] = self.load(path)

    @abstractmethod
    def load(self, path: str) -> List[KGQAData]:
        pass
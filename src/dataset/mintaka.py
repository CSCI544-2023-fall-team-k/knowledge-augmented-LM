from .kgqa_dataset import KGQADataSet
from .kgqa_data import KGQAData
from typing import List

class Mintaka(KGQADataSet):

    def load(self, path: str) -> List[KGQAData]:
        pass
from typing import List
from src.knowledge_graph import KGEntity, KGProperty


class KGQAData:
    def __init__(self, id: str, question: str, 
                 entity: KGEntity, 
                 properties: List[KGProperty],
                 sparql: str = None,
                 answers: List[KGEntity] = None) -> None:
        self.id: str = id
        self.question: str = question
        self.entity: KGEntity = entity
        self.properties: List[KGProperty] = properties
        self.sparql = sparql
        self.answers: List[KGEntity] = answers

    def __str__(self) -> str:
        return f"[{self.id}]\n" + \
            f"question: {self.question}\n" + \
            f"entity: {self.entity}\n" + \
            f"properties: {self.properties}\n" + \
            f"sparql: {self.sparql}\n" + \
            f"answers: {self.answers}"

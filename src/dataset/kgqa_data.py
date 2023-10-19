from typing import List
from src.knowledge_graph import KGEntity, KGProperty


class KGQAData:
    def __init__(self, qid: str, question: str, 
                 answers: List[KGEntity] = None) -> None:
        self.qid: str = qid
        self.question: str = question
        self.answers: List[KGEntity] = answers

    def __str__(self) -> str:
        return f"[{self.qid}]\n" + \
            f"question: {self.question}\n" + \
            f"answers: {self.answers}"

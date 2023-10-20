from sentence_transformers import SentenceTransformer, util
from typing import List

class Retriever:

    def __init__(self, model_name: str, k: int):
        self.model = SentenceTransformer(model_name)
        self.k = k

    def retrieve(self, query: str, items: List[str]) -> List[str]:
        query_embedding = self.model.encode(query)
        item_embeddings = self.model.encode(items)
        item_dict = {items[i]: item_embeddings[i] for i in range(len(items))}
        sorted_items = sorted(item_dict.items(), key=lambda x: util.dot_score(x[1], query_embedding), reverse=True)
        result = [i[0] for i in sorted_items]
        return result[:self.k]

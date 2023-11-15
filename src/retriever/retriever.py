from sentence_transformers import SentenceTransformer, util
from typing import List

class Retriever:

    def __init__(self, query_encoder: str, passage_encoder: str, k: int):
        self.query_encoder = SentenceTransformer(query_encoder)
        self.passage_encoder = SentenceTransformer(passage_encoder)
        self.k = k

    def retrieve(self, query: str, candidates: List[str]) -> List[str]:
        query_embedding = self.query_encoder.encode(query, show_progress_bar=False)
        candidate_embeddings = self.passage_encoder.encode(candidates, show_progress_bar=False)
        candidate_dict = {candidates[i]: candidate_embeddings[i] for i in range(len(candidates))}
        sorted_candidates = sorted(candidate_dict.items(), key=lambda x: util.dot_score(x[1], query_embedding), reverse=True)
        result = [i[0] for i in sorted_candidates]
        return result[:self.k]
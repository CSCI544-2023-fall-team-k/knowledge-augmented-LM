from sentence_transformers import SentenceTransformer, util
from typing import List
import logging

MODEL_NAME = "all-mpnet-base-v2"
transformer = SentenceTransformer(MODEL_NAME)

# Sort knowledges in similar order with query. 
def sort_knowledges(query: str, knowledges: List[str]) -> List[str]:
    query_embedding = transformer.encode(query)
    knowledge_embeddings = transformer.encode(knowledges)

    embedding_dict = {knowledges[i]: knowledge_embeddings[i] for i in range(len(knowledges))}
    sorted_embedding = sorted(embedding_dict.items(), key=lambda x: util.dot_score(x[1], query_embedding))
    # logging.debug((query, query_embedding))
    # logging.debug(sorted_embedding)

    return [tp[0] for tp in sorted_embedding]
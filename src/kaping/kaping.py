from src.knowledge_graph import Triple
from src.settings import Config
from src.knowledge_graph import WikiData
from src.retriever import Retriever
from .modules import GenerateAnswer, GenerateSearchQuery
import dspy

from dsp.utils import deduplicate

from typing import List
import logging

class KAPING(dspy.Module):
    def __init__(self):
        super().__init__()
        lm = dspy.OpenAI(model=Config.OPENAI_MODEL_NAME, api_key=Config.OPENAI_API_KEY, temperature=0.0, request_timeout=120)
        dspy.settings.configure(lm=lm)
        self.kg = WikiData()
        self.retriever = Retriever(model_name=Config.EMBEDDING_MODEL_NAME, k=5)
        self.generate_answer = dspy.Predict(GenerateAnswer)

        self.max_hops = 3
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(self.max_hops)]
        

    def _verbalize(self, triples: List[Triple]) -> List[str]:
        return [str((t.head.name, t.rel.name, t.tail.name)) for t in triples]
    
    def forward(self, question: str):
        logging.info(f"Question: {question}")
        # 1. Entity Linking: Extract entities in the question.
        entities = self.kg.entity_linking(question)
        logging.info(f"Entities: {entities}")
        # 2. Triple Extraction: Extract triples connected to each entity.
        matched_triples: List[Triple] = self.kg.query(entities)
        logging.info(f"Matched triples: {self._verbalize(matched_triples)}")
        # 3. Candidate Retrieval: Retrieve top-k candidates from the extracted triples using semantic similarity between the question
        candidates = self._verbalize(matched_triples)
        retrieved_triples = self.retriever.retrieve(query=question, candidates=candidates)
        logging.info(f"Retrieved triples: {retrieved_triples}")
        # 4. Answer Generation: Generate answer by prompting LLM with question and context, which is retrieved triples.
        
        #context = " ".join(retrieved_triples)
        context = retrieved_triples # unlike KAPING, generating query needs context in list form instead of verablized string
        prev_query = None
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question = question).query
            if prev_query == query:
                # print(f"Obtained identical query from hop #{hop}")
                # print("Stop")
                break
            passages = self.retriever.retrieve(query=query, candidates=candidates)
            context = deduplicate(context + passages)
            prev_query = query
        answer = self.generate_answer(question=question, context=context).answer
        return dspy.Prediction(answer=answer)




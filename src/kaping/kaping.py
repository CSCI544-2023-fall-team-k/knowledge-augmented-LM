from src.knowledge_graph import Triple
from src.settings import Config
from src.knowledge_graph import WikiData
from src.retriever import Retriever
from .modules import GenerateAnswer
import dspy
from typing import List
import logging

class KAPING(dspy.Module):
    def __init__(self):
        super().__init__()
        lm = dspy.OpenAI(model=Config.OPENAI_MODEL_NAME, api_key=Config.OPENAI_API_KEY, temperature=0.0)
        dspy.settings.configure(lm=lm)
        self.kg = WikiData()
        self.retriever = Retriever(model_name=Config.EMBEDDING_MODEL_NAME, k=5)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def _verbalize(self, triples: List[Triple]) -> List[str]:
        return [str((t.head.name, t.rel.name, t.tail.name)) for t in triples]
    
    def forward(self, question: str):
        # 1. Fetch candidate triples from KG
        logging.info(f"Question: {question}")
        entities = self.kg.entity_linking(question)
        logging.info(f"Entities: {entities}")
        matched_triples: List[Triple] = self.kg.query(entities)
        # logging.info(f"Matched triples: {self._verbalize(matched_triples)}")

        # TODO: handling the case where there is no fetched answer from KG.

        # 2. Retrieve top-k candidates by calculating embedding similarities.
        retrieved_triples = self.retriever.retrieve(query=question, items=self._verbalize(matched_triples))
        logging.info(f"Retrieved triples: {retrieved_triples}")
        context = " ".join(retrieved_triples)
        answer = self.generate_answer(question=question, context=context).answer

        return dspy.Prediction(answer=answer)
    
        # Might need dspy.Prediction when evaluation
        # return dspy.Prediction(answer=answer)



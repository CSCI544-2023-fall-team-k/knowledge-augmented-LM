from src.knowledge_graph import Triple
from src.settings import Config
from src.knowledge_graph import WikiData
from src.retriever import Retriever
from .modules import GenerateAnswer, GenerateFirstSubQuery, GenerateNextSubQuery, SolveQuestion
import dspy
import sys

from dsp.utils import deduplicate

from typing import List
import logging

class KGPPR(dspy.Module):
    def __init__(self):
        super().__init__()
        self.max_iter = 3
        self.lm = dspy.OpenAI(model=Config.OPENAI_MODEL_NAME, api_key=Config.OPENAI_API_KEY, temperature=0.1, request_timeout=30)
        dspy.settings.configure(lm=self.lm)
        self.kg = WikiData()
        self.retriever = Retriever(query_encoder=Config.QUERY_ENCODER, passage_encoder=Config.PASSAGE_ENCODER, k=5)
        self.generate_answer = dspy.Predict(GenerateAnswer)
        self.generate_first_query = dspy.ChainOfThought(GenerateFirstSubQuery)
        self.generate_next_query = dspy.ChainOfThought(GenerateNextSubQuery)
        self.solve_question = dspy.Predict(SolveQuestion)
    
    def log_history(self):
        original_stdout = sys.stdout
        with open(f'./logs.txt', 'a') as f:
            sys.stdout = f
            print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
            print(self.lm.inspect_history(n=1))
            print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
            sys.stdout = original_stdout

    def _verbalize(self, triples: List[Triple]) -> List[str]:
        return [str((t.head.name, t.rel.name, t.tail.name)) for t in triples]
    
    def forward(self, question: str):
        logging.info(f"Question: {question}")
        previous_qa = ""
        for iter in range(self.max_iter):
            if iter == 0:
                subquery = self.generate_first_query(question=question).subquery
            else:
                subquery = self.generate_next_query(question=question, previous_qa=previous_qa).subquery
            logging.info(f"[{iter}] Sub Query: {subquery}")
            #self.log_history()
            if 'NONE' in subquery: # for early stopping when we get all the information we need to solve the question
                break
            # 1. Entity Linking: Extract entities in the question.
            entities = self.kg.entity_linking(subquery)
            logging.info(f"[{iter}] Entities: {entities}")
            # 2. Triple Extraction: Extract triples connected to each entity.
            matched_triples: List[Triple] = self.kg.query(entities)
            #logging.info(f"Matched triples: {self._verbalize(matched_triples)}")
            # 3. Candidate Retrieval: Retrieve top-k candidates from the extracted triples using semantic similarity between the question
            candidates = self._verbalize(matched_triples)
            retrieved_triples = self.retriever.retrieve(query=subquery, candidates=candidates)
            logging.info(f"[{iter}] Retrieved triples: {retrieved_triples}")
            context = " ".join(retrieved_triples)
            # 4. Answer Generation: Generate answer by prompting LLM with question and context, which is retrieved triples.
            answer = self.generate_answer(question=subquery, context=context).answer
            #self.log_history()
            logging.info(f"[{iter}] Answer: {answer}")
            previous_qa += f"\n[{iter}] Subquery: {subquery}\n[{iter}] Answer: {answer}\n"
        answer = self.solve_question(question=question, previous_qa=previous_qa).answer
        #self.log_history()
        logging.info(f"Final Answer: {answer}")
        return dspy.Prediction(answer=answer)




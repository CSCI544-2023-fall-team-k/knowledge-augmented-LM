from src.knowledge_graph import WikiData, Triple, KGEntity, KGProperty
from src.dataset import KGQAData
from src.llm import LanguageModel
from src.settings import Configuration
from .embedding import sort_knowledges

from typing import List
import logging

NUM_RETRIEVED_FACTS = 10
PROMPT_INSTRUCTION = "Below are facts in the form of the triple meaningful to answer the question. Answer is between 1 to 5 words."


class KAPING:
    def __init__(self) -> None:
        self.wikidata = WikiData()
        self.llm = LanguageModel(Configuration.OPEN_AI_API_KEY, Configuration.OPEN_AI_MODEL_NAME)

    def _generate_prompt(self, verbalized_triples: List[str]) -> str:
        prompt = f"{PROMPT_INSTRUCTION}\n"
        prompt += "\n".join(verbalized_triples)
        return prompt

    def _verbalize(self, triples: List[Triple]) -> List[str]:
        return [str(t) for t in triples]
    
    def process(self, data: KGQAData) -> str:
        # 1. Fetch candidate triples from KG
        logging.info(f"Question: {data.question}")

        entities = self.wikidata.entity_linking(data.question)
        logging.info(f"Entities: {entities}")

        triples: List[Triple] = self.wikidata.query(entities)
        logging.info(f"Retrieved triples: {triples}")

        # TODO: handling the case where there is no fetched answer from KG.

        # 2. Reduce candidates by calculating embedding similarities.
        sorted_triples = sort_knowledges(data.question, self._verbalize(triples))
        logging.info(f"sorted triples: {sorted_triples}")
        final_triples = sorted_triples[::-1][: NUM_RETRIEVED_FACTS]
        logging.info(f"final triples: {final_triples}")

        # 3. Generate a question using verbalizing and prompting
        prompt = self._generate_prompt(final_triples)
        logging.info(f"prompt to ask: {prompt}")

        # 4. Ask LLM 
        response = self.llm.ask(prompt)
        logging.info(f"response from llm: {response}")
        return response
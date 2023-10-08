from src.knowledge_graph import WikiData, Triple, KGEntity, KGProperty
from src.dataset import KGQAData
from src.llm import LanguageModel
from src.settings import Configuration
from .embedding import sort_knowledges

from typing import List
import logging

NUM_RETRIEVED_FACTS = 10
PROMPT_INSTRUCTION = "Below are facts in the form of the triple meaningful to answer the question."


class Kaping:
    def __init__(self) -> None:
        self.wikidata = WikiData()
        self.llm = LanguageModel(Configuration.OPEN_AI_API_KEY, Configuration.OPEN_AI_MODEL_NAME)

    def _generate_prompt(self, verbalized_triples: List[str]) -> str:
        prompt = f"{PROMPT_INSTRUCTION}\n"
        prompt += "\n".join(verbalized_triples)
        return prompt

    def _verbalize(self, triples: List[Triple]) -> List[str]:
        return [str(t) for t in triples]
    
    def _build_triples(self, entity: KGEntity, properties: List[KGProperty], knowledges: List[dict]) -> List[Triple]:
        triples = []
        for p in properties:
            for k in knowledges:
                k_name = k["xLabel"]["value"]
                triples.append(Triple(entity, p, KGEntity(k_name)))

        return triples

    def process(self, data: KGQAData) -> str:
        # 1. Fetch candidate triples from KG
        logging.debug(f"process question: {data.question}")
        knowledges: List[dict] = self.wikidata.query(data.sparql)
        # TODO: handling the case where there is no fetched answer from KG.
        triples: List[Triple] = self._build_triples(data.entity, data.properties, knowledges)
        logging.debug(f"retrieved triples: {triples}")

        # 2. Reduce candidates by calculating embedding similarities.
        sorted_triples = sort_knowledges(data.question, self._verbalize(triples))
        logging.debug(f"sorted triples: {sorted_triples}")
        final_triples = sorted_triples[::-1][: NUM_RETRIEVED_FACTS]
        logging.debug(f"final triples: {final_triples}")

        # 3. Generate a question using verbalizing and prompting
        prompt = self._generate_prompt(final_triples)
        logging.debug(f"prompt to ask: {prompt}")

        # 4. Ask LLM 
        response = self.llm.ask(prompt)
        logging.debug(f"response from llm: {response}")
        return response
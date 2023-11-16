from src.knowledge_graph import KGEntity, KGProperty
from .kgqa_dataset import KGQADataSet
from .kgqa_data import KGQAData
from typing import List
import logging
import json

class ComplexWebQ(KGQADataSet):

    def load(self, path: str) -> List[KGQAData]:
        datasets: List[KGQAData] = []

        with open(path, encoding='utf-8') as f:
            json_dict = json.load(f)
            for complex_web_q_data in json_dict:
                question_id = complex_web_q_data["ID"]
                raw_question = complex_web_q_data["question"]
                                
                answer_data = complex_web_q_data["answers"]
                answers = []
                if answer_data:
                    for answer in answer_data:
                        answers.append(KGEntity(answer["answer"]))
                        # Assume that aliases are also answers.
                        aliases = answer["aliases"]
                        for alias in aliases:
                            answers.append(KGEntity(alias))
                else:
                    continue
                datasets.append(KGQAData(question_id, raw_question, answers))
                
        logging.info(f"number of parsed questions: {len(datasets)}")
        
        return datasets
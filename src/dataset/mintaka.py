from src.knowledge_graph import KGEntity, KGProperty
from .kgqa_dataset import KGQADataSet
from .kgqa_data import KGQAData
from typing import List
import logging
import json

class Mintaka(KGQADataSet):

    def load(self, path: str) -> List[KGQAData]:
        datasets: List[KGQAData] = []

        with open(path, encoding='utf-8') as f:
            json_dict = json.load(f)
            for mintaka_data in json_dict:
                question_id = mintaka_data["id"]
                raw_question = mintaka_data["question"]
                                
                answer_data = mintaka_data["answer"]["answer"]                
                answers = [KGEntity(answer["label"]["en"]) if type(answer) is dict else str(answer) for answer in answer_data] if answer_data else []
                
                datasets.append(KGQAData(question_id, raw_question, answers))

        logging.info(f"number of parsed questions: {len(datasets)}")
        
        return datasets
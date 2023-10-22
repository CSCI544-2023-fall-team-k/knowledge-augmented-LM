from src.knowledge_graph import KGEntity, KGProperty
from .kgqa_dataset import KGQADataSet
from .kgqa_data import KGQAData
from typing import List
import logging
import json

class WebQSP(KGQADataSet):

    def load(self, path: str) -> List[KGQAData]:
        datasets: List[KGQAData] = []

        with open(path) as f:
            json_dict = json.load(f)
            for question in json_dict["Questions"]:
                question_id = question["QuestionId"]
                raw_question = question["RawQuestion"]

                # Use first parsed query only.
                parse = question["Parses"][0]           
                answers = [KGEntity(answer["EntityName"]) for answer in parse["Answers"]] if parse["Answers"] else []
                datasets.append(KGQAData(question_id, raw_question, answers))

        logging.info(f"number of parsed questions: {len(datasets)}")
        return datasets
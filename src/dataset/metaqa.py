from src.knowledge_graph import KGEntity, KGProperty
from .kgqa_dataset import KGQADataSet
from .kgqa_data import KGQAData
from typing import List
import logging

class MetaQA(KGQADataSet):

    def load(self, path: str) -> List[KGQAData]:
        datasets: List[KGQAData] = []

        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            for id, line in enumerate(lines):
                qa = line.split("\t")
                quesiton_id = id
                question, answers = qa[0].split(" "), qa[1]
                for i, word in enumerate(question):
                    if "[" in word:
                        question[i] = question[i].replace("[","")
                    if "]" in word:
                        question[i] = question[i].replace("]","")
                question = ' '.join(question)
                kg_entity_answers = []
                for answer in answers.split("|"):
                    answer = answer.replace('\n','')
                    kg_entity_answers.append(KGEntity(answer))
                datasets.append(KGQAData(quesiton_id, question, kg_entity_answers))
        logging.info(f"number of parsed questions: {len(datasets)}")
        
        return datasets
from kaping import KAPING
from src.dataset import WebQSP
from src.dataset import Mintaka
from src.dataset import ComplexWebQ
from src.dataset import MetaQA
from kaping.metrics import exact_matching
from dspy.primitives import Example
from kaping.evaluate import Evaluate
import argparse
import logging
import os
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(data: str, outfile: str = "evaluation_result.csv", num_test: int = 500):

    if data.lower() == "WebQSP".lower():
        dataset = WebQSP(path="resources/WebQSP/data/WebQSP.test.processed.json")
        logging.info(f"data = WebQSP")
    elif data.lower() == "mintaka".lower():
        dataset = Mintaka(path="resources/mintaka/data/mintaka_test.json")
        logging.info(f"data = mintaka")
    elif data.lower() == "ComplexWebQ".lower():
        dataset = ComplexWebQ(path="resources/ComplexWebQuestions/ComplexWebQuestions_train.json")
        logging.info(f"data = ComplexWebQ")
    elif data.lower() == "MetaQA_1-hop".lower():
        dataset = MetaQA(path="resources/MetaQA/vanilla_1-hop/qa_test.txt")
        logging.info(f"data = MetaQA_1-hop")
    elif data.lower() == "MetaQA_2-hop".lower():
        dataset = MetaQA(path="resources/MetaQA/vanilla_2-hop/qa_test.txt")
        logging.info(f"data = MetaQA_2-hop")
    elif data.lower() == "MetaQA_3-hop".lower():
        dataset = MetaQA(path="resources/MetaQA/vanilla_3-hop/qa_test.txt")
        logging.info(f"data = MetaQA_3-hop")
    else:
        logging.info(f"Wrong data! It should be one of them. - 'WebQSP', 'mintaka', 'ComplexWebQ', 'MetaQA_1-hop', 'MetaQA_2-hop', 'MetaQA_3-hop'")
        return
        
    logging.info(f"Num questions: {len(dataset.data)}")

    kaping = KAPING()
    dspy_dataset = []

    seed_value = 40

    random.seed(seed_value)
    random_indexes = random.sample(range(len(dataset.data)), num_test)

    for i in random_indexes:
        data = dataset.data[i]
        dspy_dataset.append(Example({'question':data.question, 'answer':[a.name if hasattr(a, 'name') else a for a in data.answers]}))
    dspy_dataset = [x.with_inputs('question') for x in dspy_dataset]

    evaluate_on_gts = Evaluate(devset=dspy_dataset, outfile=outfile, num_threads=1, display_progress=True)
    evaluation_result = evaluate_on_gts(kaping, metric=exact_matching)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(prog='Knowledge Augmented Language Model')
    parser.add_argument("--data", choices=['WebQSP', 'mintaka', 'ComplexWebQ', 'MetaQA_1-hop', 'MetaQA_2-hop', 'MetaQA_3-hop'], default='ComplexWebQ')
    parser.add_argument("--outfile", type=str, default="evaluation_result.csv")
    parser.add_argument("--num_test", type=int, default=10)

    ####################
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('logfile.log')
    stream_handler = logging.StreamHandler()

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    ####################

    args = parser.parse_args()
    logging.root.setLevel(logging.INFO)
    main(args.data, args.outfile, args.num_test)
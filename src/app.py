from kaping import KAPING
from src.dataset import WebQSP
from src.dataset import Mintaka
from kaping.metrics import exact_matching
from dspy.primitives import Example
from kaping.evaluate import Evaluate
import argparse
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(data: str, outfile: str = "evaluation_result.csv"):

    if data.lower() == "WebQSP".lower():
        dataset = WebQSP(path="resources/WebQSP/data/WebQSP.test.processed.json")
        logging.info(f"data = WebQSP")
    elif data.lower() == "mintaka".lower():
        dataset = Mintaka(path="resources/mintaka/data/mintaka_test.json")
        logging.info(f"data = mintaka")
    else:
        logging.info(f"Wrong data! It should be 'WebQSP' or 'mintaka'.")
        return
        
    logging.info(f"Num questions: {len(dataset.data)}")

    kaping = KAPING()
    dspy_dataset = []

    for data in dataset.data[:3]:
        dspy_dataset.append(Example({'question':data.question, 'answer':[a.name if hasattr(a, 'name') else a for a in data.answers]}))
    dspy_dataset = [x.with_inputs('question') for x in dspy_dataset]
   

    # 3. Evaluate the results
    # logging.info(f"Evaluate the results")
    # TODO
    
    evaluate_on_gts = Evaluate(devset=dspy_dataset, outfile=outfile, num_threads=1, display_progress=True)
    evaluation_result = evaluate_on_gts(kaping, metric=exact_matching)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(prog='Knowledge Augmented Language Model')
    parser.add_argument("--data", choices=['WebQSP', 'mintaka'], default='WebQSP')
    parser.add_argument("--outfile", type=str, default="evaluation_result.csv")

    args = parser.parse_args()
    logging.root.setLevel(logging.INFO)
    main(args.data, args.outfile)
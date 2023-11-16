from kgppr import KGPPR
from src.dataset import WebQSP
from src.dataset import Mintaka
from src.dataset import ComplexWebQ
from kgppr.metrics import exact_matching
from dspy.primitives import Example
from kgppr.evaluate import Evaluate
import argparse
import logging
import os
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
    else:
        logging.info(f"Wrong data! It should be 'WebQSP' or 'mintaka'.")
        return
        
    logging.info(f"Num questions: {len(dataset.data)}")

    kgppr = KGPPR()
    dspy_dataset = []

    for data in dataset.data[:num_test]:
        dspy_dataset.append(Example({'question':data.question, 'answer':[a.name if hasattr(a, 'name') else a for a in data.answers]}))
    dspy_dataset = [x.with_inputs('question') for x in dspy_dataset]
   

    # 3. Evaluate the results
    # logging.info(f"Evaluate the results")
    # TODO
    
    evaluate_on_gts = Evaluate(devset=dspy_dataset, outfile=outfile, num_threads=1, display_progress=True)
    evaluation_result = evaluate_on_gts(kgppr, metric=exact_matching)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(prog='Knowledge Augmented Language Model')
    parser.add_argument("--data", choices=['WebQSP', 'mintaka', 'ComplexWebQ'], default='WebQSP')
    parser.add_argument("--outfile", type=str, default="evaluation_result.csv")
    parser.add_argument("--num_test", type=int, default=3)

    args = parser.parse_args()
    logging.root.setLevel(logging.INFO)
    main(args.data, args.outfile, args.num_test)
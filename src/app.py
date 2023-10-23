from kaping import KAPING
from src.dataset import WebQSP
from src.dataset import Mintaka
import argparse
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(filename: str, datatype: str, outfile: str = "out.txt"):
    logging.info(f"Read dataset : {filename}")
    
    if datatype.lower() == "WebQSP".lower():
        inputdata = WebQSP(path=filename)
        logging.info(f"data type = WebQSP")
    elif datatype.lower() == "mintaka".lower():
        inputdata = Mintaka(path=filename)
        logging.info(f"data type = mintaka")
    else:
        logging.info(f"Wrong data type! Datatype should be 'WebQSP' or 'mintaka'.")
        return
        
    logging.info(f"Num questions: {len(inputdata.data)}")

    kaping = KAPING()

    with open(outfile, 'w') as f:
        datasets = inputdata.data[:10]
        for data in datasets:
            # TODO: insert delay between each call(RateLimitError: Rate limit reached for default-gpt-3.5-turbo in organization org-esrvXzMVq3Ig0JF7IPOfPHmb on requests per min. Limit: 3 / min.)
            prediction = kaping(data.question)  
            logging.info(f"Prediction: {prediction}, Gold Answer: {[a.name if hasattr(a, 'name') else a for a in data.answers]}")
            f.write(f"{data.qid}\t{data.question}\t{prediction}\n")

    # 3. Evaluate the results
    # logging.info(f"Evaluate the results")
    # TODO


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(prog='Knowledge Augmented Language Model')
    parser.add_argument("filename")
    parser.add_argument("datatype", choices=['WebQSP', 'mintaka'])
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--outfile", type=str, default="out.txt")

    args = parser.parse_args()
    logging.root.setLevel(logging.INFO)
    main(args.filename, args.datatype, args.outfile)
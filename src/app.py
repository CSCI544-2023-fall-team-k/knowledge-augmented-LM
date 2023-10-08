from kaping import Kaping
from src.dataset import WebQSP
import argparse
import logging

def main(filename: str, head: int = None, outfile: str = "out.txt"):
    # 1. Read dataset
    logging.info(f"Read dataset : {filename}")
    web_qsp = WebQSP(path=filename)
    logging.info(f"Num questions: {len(web_qsp.data)}")

    # 2. Process KAPING per each question data
    logging.info(f"Start processing data")
    answers = []
    app = Kaping()

    with open(outfile, 'w') as f:
        datasets = web_qsp.data if head is None else web_qsp.data[:head]
        for data in datasets:
            # TODO: insert delay between each call(RateLimitError: Rate limit reached for default-gpt-3.5-turbo in organization org-esrvXzMVq3Ig0JF7IPOfPHmb on requests per min. Limit: 3 / min.)
            answer = app.process(data)  
            answers.append(answer)
            f.write(f"{data.id}\t{data.question}\t{answer}\n")

    # 3. Evaluate the results
    logging.info(f"Evaluate the results")
    # TODO


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Knowledge Augmented Language Model')
    parser.add_argument("filename")
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--outfile", type=str, default="out.txt")

    args = parser.parse_args()
    main(args.filename, args.head, args.outfile)
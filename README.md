# Knowledge Augmented Language Model 
This repository documents the implementation of the referenced paper and subsequent research for improvement.

# Reference
- [Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering](https://browse.arxiv.org/pdf/2306.04136.pdf), by Jinheon Baek1, Alham Fikri Aji, Amir Saffari


# How to run
1. Execution environment

    - This project use [poetry](https://python-poetry.org/) to manage and install python dependencies. 
    - To install all the prerequisite packages at once, run `poetry install` in the root directory.
    - To run the project program within the pre-defined environment, run `poetry run python ${PROGRAM_NAME}`.
    - To activate the virtual environment in a shell, run `poetry shell`.

2. Run
    ```sh
    > python src/app.py ${IN_FILE_PATH_NAME} --head ${NUM_QUESTIONS} --outfile ${OUT_FILE_PATH_NAME}
    ```
    - `IN_FILE_PATH_NAME` : Path and name of an input file. The input file should be Json WebQSP datasets with SPARQL queries processed for Wikidata. Original WebQSP datasets contained queries for Freebase, requiring further preprocessing for entity/property mapping. 
      - Preprocessing script: `resources/preprocess.py`.
    - `NUM_QUESTIONS` : The number of questions needed to be processed from the beginning. Default value is `None`, processing all the questions from the input file. 
    - `OUT_FILE_PATH_NAME` : Path and name of an output file. Default value is `./out.txt`

    - example
      ```sh
      > python src/app.py resources/WebQSP/data/WebQSP.train.processed.json --head 10
      ```

# Knowledge Graph Prompting using Procedural Reasoning
This repository contains implementations on KAPPR. This work is done in CSCI544 Applied Natural Language Processing by Prof. Mohammad Rostami on Fall 2023.

# Motivation
Although Retreival Augmented Generation (RAG) has become the standard of all the knowledge-augmented language modeling tasks, two main limitations exist. First, it requires massive data to train both the retriever and LM. Especially, fine-tuning pre-trained LM costs a lot of resources and is not desirable in real-world settings. Second, the method tries to solve questions in one-shot, where knowledge retriever and LM are used only once to solve the given question. However, for complex questions requiring multiple reasoning steps, a one-shot approach may be insufficient to provide accurate answers. 

To mitigate these limitations, we propose Knowledge Graph Prompting using Procedural Reasoning (KGPPR), which is a zero-shot LM prompting framework that uses procedural reasoning to solve complex knowledge graph based questions. Specifically, to address the mentioned limitations, KGPPR employs two modules.

1. Zero-Shot KG Prompting: Similar to RAG, the method adopts both knowledge graph retriever and LM to solve a question. First, it retrieves top-K knowledge graph triples that are relevant to the question. The retrieved knowledge graph triples are then converted into natural language and used as prompts for LM.

2. Procedural Reasoning: The method employs multiple rounds of reasoning steps to solve a question. For each round, it uses chain-of-thought (CoT) to generate the next sub-question that needs to be addressed in a step-by-step fashion. Finally, the sub-question is solved using the Zero-Shot KG Prompting. For the next round, we utilize previous answers to generate answers for the next round. 

# How to run
1. Execution environment

    - This project use [poetry](https://python-poetry.org/) to manage and install python dependencies. 
    - To install all the prerequisite packages at once, run `poetry install` in the root directory.
    - To run the project program within the pre-defined environment, run `poetry run python ${PROGRAM_NAME}`.
    - To activate the virtual environment in a shell, run `poetry shell`.

2. Configuration
    - In order to use the OpenAI API, ensure that your API key is added to the `src/settings/config.py` file prior to running the program.
        ```python
        class Configuration:
            OPEN_AI_API_KEY = "${KEY_SHOULD_BE_ADDED}"
            OPEN_AI_MODEL_NAME = "gpt-3.5-turbo"
        ```

3. Run
    ```sh
    python src/app.py ${IN_FILE_PATH_NAME} --head ${NUM_QUESTIONS} --outfile ${OUT_FILE_PATH_NAME}
    ```
    - `IN_FILE_PATH_NAME` : Path and name of an input file. The input file should be Json WebQSP datasets with SPARQL queries processed for Wikidata. Original WebQSP datasets contained queries for Freebase, requiring further preprocessing for entity/property mapping. 
      - Preprocessing script: `resources/preprocess.py`.
    - `NUM_QUESTIONS` : The number of questions needed to be processed from the beginning. Default value is `None`, processing all the questions from the input file. 
    - `OUT_FILE_PATH_NAME` : Path and name of an output file. Default value is `./out.txt`

    - example
      ```sh
      python src/app.py resources/WebQSP/data/WebQSP.train.processed.json --head 10
      ```
# Reference
- [Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering](https://browse.arxiv.org/pdf/2306.04136.pdf), by Jinheon Baek1, Alham Fikri Aji, Amir Saffari

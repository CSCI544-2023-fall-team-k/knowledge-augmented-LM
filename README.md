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

2. Configuration
    - In order to use the OpenAI API, ensure that your API key is added to the `src/settings/config.py` file prior to running the program.
        ```python
        class Configuration:
            OPENAI_API_KEY = "${KEY_SHOULD_BE_ADDED}"
            OPENAI_MODEL_NAME = "gpt-3.5-turbo"
            EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
        ```

3. Run
    ```sh
    python src/app.py --data ${DATA} --outfile ${OUT_FILE_PATH_NAME}
    ```
    - `DATA` : The dataset you will use. Two options: 'WebQSP' or 'mintaka'.
    - `OUT_FILE_NAME` : The name of the output file. Default value is `evaluation_resultc.csv`

    - example
      ```sh
      python src/app.py --data WebQSP --outfile evaluation_resultc.csv
      ```

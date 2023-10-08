import unittest
import logging
from src.llm import LanguageModel
from src.settings import Configuration

logging.basicConfig(level=logging.DEBUG)

class TestLanguageModel(unittest.TestCase):

    def test_ask(self):
        api_key = Configuration.OPEN_AI_API_KEY
        model_name = Configuration.OPEN_AI_MODEL_NAME

        llm = LanguageModel(api_key, model_name)
        prompt = "what is the name of justin bieber brother?"
        reply = llm.ask(prompt)
        print(reply)
        self.assertGreater(len(reply), 0)

if __name__ == '__main__':
    unittest.main()
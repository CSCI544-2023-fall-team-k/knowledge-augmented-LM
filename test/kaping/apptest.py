import unittest
import logging
from src.dataset import WebQSP
from src.kaping import Kaping

logging.basicConfig(level=logging.DEBUG)

class TestKapingApp(unittest.TestCase):

    def test_query(self):
        train_dataset = WebQSP("./resources/WebQSP/data/WebQSP.train.processed.json")
        app = Kaping()
        app.process(train_dataset.data[4])
        
if __name__ == '__main__':
    unittest.main()
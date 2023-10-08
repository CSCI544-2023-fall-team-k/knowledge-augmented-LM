import unittest
import logging
from src.knowledge_graph import WikiData

logging.basicConfig(level=logging.DEBUG)

class TestWikiData(unittest.TestCase):

    def test_query(self):
        sparql = "SELECT DISTINCT ?xLabel WHERE { FILTER (?x != wd:Q352) ?x rdfs:label ?xLabel filter (lang(?xLabel) = 'en'). wd:Q352 wdt:P20 ?x . } "
        response = WikiData().query(sparql)
        self.assertGreater(len(response), 0)
        
if __name__ == '__main__':
    unittest.main()
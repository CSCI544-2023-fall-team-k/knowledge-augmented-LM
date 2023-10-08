import unittest
from src.dataset import *

class TestWebQSP(unittest.TestCase):

    def test_initialize(self):
        train_dataset = WebQSP("./resources/WebQSP/data/WebQSP.train.json")
        self.assertGreater(len(train_dataset.data), 0)
        sample_data = train_dataset.data[0]
        print(sample_data)
        self.assertEqual(sample_data.id, "WebQTrn-0")
        self.assertEqual(sample_data.question,  "what is the name of justin bieber brother?")
        self.assertEqual(sample_data.sparql, "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.06w2sn5)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.06w2sn5 ns:people.person.sibling_s ?y .\n?y ns:people.sibling_relationship.sibling ?x .\n?x ns:people.person.gender ns:m.05zppz .\n}\n")
        self.assertGreater(len(sample_data.answers), 0)

        sample_answer = sample_data.answers[0]
        self.assertEqual(sample_data.entity.name, "Justin Bieber")
        self.assertEqual(sample_answer.name, "Jaxon Bieber")


if __name__ == '__main__':
    unittest.main()
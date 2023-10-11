from .wikidata import WikiData

class KnowledgeRetriever:

    def __init__(self) -> None:
        self._wikidata = WikiData()
        # TODO: implement wrapper class
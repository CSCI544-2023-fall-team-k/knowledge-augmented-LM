import requests
from urllib.parse import quote
from typing import List
import logging

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

class WikiData:

    def query(self, query: str) -> List[dict]:
        headers = { 
            "Accept": "application/sparql-results+json"
        }
        encoded_query = quote(query, safe="()*!\'")
        response = requests.get(SPARQL_ENDPOINT + f"?query={encoded_query}", headers=headers)
        logging.debug(response.json())

        return response.json()['results']['bindings']
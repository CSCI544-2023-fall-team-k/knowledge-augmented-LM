import requests
from urllib.parse import quote
from typing import List
import logging
from string import Template
from .data import KGEntity, KGProperty, Triple
from wikidataintegrator import wdi_core
import spacy 

sparql_query_temp = Template("""SELECT ?rel_id ?rel_label ?tail_id ?tail_label ?tail_desc WHERE {wd:$entity ?rel_id ?tail_id.
FILTER (LANG(?tail_label) = 'en').
  ?property ?ref ?rel_id;     
  rdf:type wikibase:Property;   
  SERVICE wikibase:label {
    ?tail_id rdfs:label ?tail_label .  
    ?tail_id schema:description ?tail_desc.    
    ?property rdfs:label ?rel_label.    
    bd:serviceParam wikibase:language 'en'.   
  }       
}
""")

class WikiData:

    def __init__(self):
        self.entity_linker = spacy.load("en_core_web_md")
        self.entity_linker.add_pipe("entityLinker", last=True)

    def query(self, entities: List[str]) -> List[dict]:
        matched_triples = []
        for head_id, head_label in entities:
            head_id = 'Q'+str(head_id)
            sparql_query = sparql_query_temp.substitute(entity=head_id)
            # TODO: might want to replace wdi_core with the simple requests module
            result_df = wdi_core.WDFunctionsEngine.execute_sparql_query(sparql_query, as_dataframe=True)
            for _, row in result_df.iterrows():
                matched_triples.append(Triple(KGEntity(head_label), KGProperty(row['rel_label']), KGEntity(row['tail_label'])))
        return matched_triples
    
    def entity_linking(self, question: str):
        doc = self.entity_linker(question)
        entities = []
        for sent in doc.sents:
            for ent in sent._.linkedEntities:
                entities.append((ent.get_id(), ent.get_label()))
        return entities

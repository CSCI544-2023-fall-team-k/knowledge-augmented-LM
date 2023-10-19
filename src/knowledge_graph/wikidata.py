import requests
from urllib.parse import quote
from typing import List
import logging
from wikidataintegrator import wdi_core
import spacy 

SPARQL_QUERY = """SELECT ?rel_id ?rel_label ?tail_id ?tail_label ?tail_desc
WHERE {   
  wd:{entity} ?rel_id ?tail_id.
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
"""

class WikiData:

    def __init__(self):
        self.entity_linker = spacy.load("en_core_web_md")
        self.entity_linker.add_pipe("entityLinker", last=True)


    def query(self, entities: List[str]) -> List[dict]:
        for head_id, head_label in entities:
            sparql_query = SPARQL_QUERY.format(entity=head_id)
            result_df = wdi_core.WDFunctionsEngine.execute_sparql_query(sparql_query, as_dataframe=True)
            #result_df = result_df[['rel_label','tail_label']]
            #result_df['head_label'] = [head_label]*len(result_df)
            # TODO: return [Entity, Property, Entity]
        #return result_df
    
    def entity_linker(self, question: str):
        doc = self.entity_linker(question)
        entities = []
        for sent in doc.sents:
            sent._.linkedEntities.pretty_print()
            for ent in sent._.linkedEntities:
                entities.append((ent.get_id(), ent.get_label()))
        return entities

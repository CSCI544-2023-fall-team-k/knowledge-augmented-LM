# Convert freebase SparQL to wikidata one
import json
import re

# Need freebase-wikidata-convert: https://github.com/happen2me/freebase-wikidata-convert
from converter import EntityConverter, PropertyConverter

entity_converter = EntityConverter("https://query.wikidata.org/sparql")
property_converter = PropertyConverter("https://www.wikidata.org/wiki/Wikidata:WikiProject_Freebase/Mapping")


# SPARQL does not have or, and operators. 
REPLACE_TOKENS = {
    "\n" : " ",
    " OR " : "||", 
    " AND " : "&&",
}

def replace_token(sparql: str) -> str:
    for token in REPLACE_TOKENS.keys():
        sparql = sparql.replace(token, REPLACE_TOKENS[token])
    return sparql

def convert_entity_id(entity_id: str) -> str:
    entity_id = entity_id.replace("ns:", "")
    processed_id = "/" + entity_id.replace(".", "/")
    wikidata_id = entity_converter.get_wikidata_id(processed_id)
    return f"wd:{wikidata_id}" if wikidata_id else None

def convert_entity_property(entity_property: str) -> str:
    entity_property = entity_property.replace("ns:", "")
    processed_property = "/" + entity_property.replace(".", "/")
    wikidata_property = property_converter.get_wikidata_property(processed_property)
    return f"wdt:{wikidata_property}" if wikidata_property else None

input_path = "./WebQSP/data/WebQSP.train.json"
output_path = "WebQSP.train.processed.json"

with open(input_path) as f:
    data = json.load(f)
    dropped = [False] * len(data["Questions"])
    for i, question in enumerate(data["Questions"]):
        print(f"process question {i}")
        parse = question["Parses"][0]
        sparql = parse["Sparql"].replace("PREFIX ns: <http://rdf.freebase.com/ns/>\n", "")
        sparql = sparql.replace("SELECT DISTINCT ?x", "SELECT DISTINCT ?xLabel")
        sparql = sparql.replace("FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))", "?x rdfs:label ?xLabel filter (lang(?xLabel) = 'en').")
        
        tokens = re.findall(r"(ns:[\w.]+)", sparql)
        for token in tokens:
            if token.startswith("ns:m."):
                wikidata_id = convert_entity_id(token)
                if wikidata_id is None:
                    dropped[i] = True
                    break
                sparql = sparql.replace(token, wikidata_id)

            elif token.startswith("ns:"):
                wikidata_property = convert_entity_property(token)
                if wikidata_property is None:
                    dropped[i] = True
                    break
                sparql = sparql.replace(token, wikidata_property)

        if dropped[i]:
            continue

        sparql = replace_token(sparql)
        sparql = sparql
        parse["Sparql"] = sparql

    data["Questions"] = [x for i, x in enumerate(data["Questions"]) if not dropped[i]]
    print(f"Skipped {dropped.count(True)} out of {len(dropped)}.")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
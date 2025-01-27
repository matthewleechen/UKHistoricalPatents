######## run named entity recognition ###########                    
##
## the code requires a folder structure like the following:
##
## current_working_dir/
## └── input_folder/
##     ├── 1617.json
##     ├── ...
##     ├── 1898.json
##     └── 1899.json
##
## where each JSON file contains:
## [
##   {
##     "patent_id": "GB...",
##     "full_text": [
##       {"page_num": 1, "page_text": "..."},
##       {"page_num": 2, "page_text": "..."},
##       ...
##     ],
##     ... (other fields)
##   },
##   ... (more patents)
## ]
##
##
## the code creates entities.json which contains "patent_id", "year" and
## a "front_page_entities" key to each patent dictionary, 
## containing a list of named entities found on the first page of the patent.
##
## each entity is represented as:
## {
##   "class": "ENTITY_TYPE",
##   "entity_text": "Extracted entity text",
##   "start": start_index,
##   "end": end_index
## }
##
## The script processes only the first page (page with the smallest page_num) 
## of each patent for named entity recognition
##
##

import os
import json
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from datasets import Dataset
import string
import textdistance


model_name = 'matthewleechen/patent_entities_ner'
inf_batch_size = 64 
folder_path = "gbpatentdata_OCR_output"


# load saved model weights + tokenizer
def load_model():
    
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer




def custom_recognizer(text, model, tokenizer, device=0):

    # HF ner pipeline
    token_level_results = pipeline("ner", model=model, device=0, tokenizer=tokenizer)(text)

    # keep entities tracked
    entities = []
    current_entity = None

    for item in token_level_results:

        tag = item['entity']

        # replace '▁' with space for easier reading (_ is created by the XLM-RoBERTa tokenizer)
        word = item['word'].replace('▁', ' ')

        # aggregate I-O-B tagged entities
        if tag.startswith('B-'):

            if current_entity:
                entities.append(current_entity)

            current_entity = {'type': tag[2:], 'text': word.strip(), 'start': item['start'], 'end': item['end']}

        elif tag.startswith('I-'):

            if current_entity and tag[2:] == current_entity['type']:
                current_entity['text'] += word
                current_entity['end'] = item['end']

            else:

                if current_entity:
                    entities.append(current_entity)

                current_entity = {'type': tag[2:], 'text': word.strip(), 'start': item['start'], 'end': item['end']}

        else:
            # deal with O tag
            if current_entity:
                entities.append(current_entity)
            current_entity = None

    if current_entity:
        # add to entities
        entities.append(current_entity)

    # track entity merges
    merged_entities = []

    # merge entities of the same type
    for entity in entities:
        if merged_entities and merged_entities[-1]['type'] == entity['type'] and merged_entities[-1]['end'] == entity['start']:
            merged_entities[-1]['text'] += entity['text']
            merged_entities[-1]['end'] = entity['end']
        else:
            merged_entities.append(entity)

    # clean up extra spaces
    for entity in merged_entities:
        entity['text'] = ' '.join(entity['text'].split())

    # convert to list of dicts
    return [{'class': entity['type'],
             'entity_text': entity['text'],
             'start': entity['start'],
             'end': entity['end']} for entity in merged_entities]





def process_entities(entities):
    
    processed_entities = []
    
    for entity in entities:
        
        # drop whitespace and punctuation
        entity_text_clean = entity['entity_text'].strip().translate(str.maketrans('', '', string.punctuation))
        
        # remove entities with entity_text length <=2
        if len(entity_text_clean) > 2:
            
            processed_entities.append(entity)
            
    # deduplicate exact matches on class and entity_text
    deduped_entities = []
    
    seen = set()
    
    for entity in processed_entities:
        
        class_text = entity['class'].lower().replace(' ', '')
        
        entity_text_norm = entity['entity_text'].lower().replace(' ', '')
        
        key = (class_text, entity_text_norm)
        
        if key not in seen:
            
            seen.add(key)
            
            deduped_entities.append(entity)
            
    # deduplicate on same class and similar entity_text using Jaro-Winkler distance, excluding 'DATE' class
    final_entities = []
    
    for entity in deduped_entities:
        
        class_text = entity['class'].lower().replace(' ', '')
        
        entity_text = entity['entity_text']
        
        is_duplicate = False
        
        # skip Jaro-Winkler deduplication for 'DATE' class
        if class_text == 'DATE':
            
            final_entities.append(entity)
            
            continue
            
        for kept_entity in final_entities:
            
            kept_class_text = kept_entity['class'].lower().replace(' ', '')
            
            if class_text == kept_class_text:
                
                similarity = textdistance.jaro_winkler.normalized_similarity(
                    entity_text.lower(), kept_entity['entity_text'].lower()
                )
                
                # set threshold of 0.9
                if similarity >= 0.9 and entity_text.lower() != kept_entity['entity_text'].lower():
                    
                    is_duplicate = True
                    
                    break
                    
        if not is_duplicate:
            
            final_entities.append(entity)
            
    return final_entities




def process_json_file(file_path, model, tokenizer):
    
    year = int(os.path.splitext(os.path.basename(file_path))[0])
    
    with open(file_path, 'r') as f:
        
        data = json.load(f)

    
    processed_data = []
    
    for item in data:
        
        patent_id = item['patent_id']
        
        full_text = item['full_text']
        
        # Find the page with the smallest page_num
        first_page = min(full_text, key=lambda x: x['page_num'])
        
        if first_page and first_page['page_text'].strip():
            
            processed_data.append({
                'patent_id': patent_id,
                'year': year,
                'page_text': first_page['page_text']
            })
    
    # create dataset
    dataset = Dataset.from_list(processed_data)
    
    results = []
    
    # collect results using custom recognizer and apply post-processing
    for text in tqdm(dataset['page_text']):
        
        entities = custom_recognizer(text, model, tokenizer)

        entities = process_entities(entities)
        
        results.append(entities)
    
    # add results to dataset
    dataset = dataset.add_column('front_page_entities', results)
    
    # gather only the desired fields for entities.json
    new_entries = []
    for item, entities in zip(data, dataset['front_page_entities']):
        new_entries.append({
            'patent_id': item['patent_id'],
            'year': year,
            'front_page_entities': entities
        })
        
    # read existing entities.json or create a new list if none exists
    if os.path.exists('entities.json'):
        with open('entities.json', 'r') as ef:
            existing_data = json.load(ef)
    else:
        existing_data = []
        
    # append new entries and write them to entities.json
    existing_data.extend(new_entries)
    with open('entities.json', 'w') as ef:
        json.dump(existing_data, ef, indent=2)
    
    print(f"appended named entities to entities.json from: {os.path.basename(file_path)}")


def process_json_files(folder_path, model, tokenizer):
    
    for filename in os.listdir(folder_path):
        
        if filename.endswith('.json'):
            
            file_path = os.path.join(folder_path, filename)
            
            process_json_file(file_path, model, tokenizer)



model, tokenizer = load_model()
process_json_files(folder_path, model, tokenizer)
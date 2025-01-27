######## run title extraction ###########                    
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
## the code adds a "patent_title" key to each patent dictionary, 
## containing the extracted title from the first page of the patent.
##
## if multiple title entities are found, only keeps distinct ones
## with Jaro-Winkler similarity < 0.95
##
## The script processes only the first page (page with the smallest page_num) 
## of each patent for title extraction
##
##

import os
import json
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from datasets import Dataset
import textdistance


model_name = 'gbpatentdata/patent_titles_ner'
inf_batch_size = 64 
folder_path = "300YearsOfBritishPatents"


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


def filter_similar_titles(title_entities):
    
    if not title_entities:
        return []
    
    filtered_titles = [title_entities[0]]
    
    # compare each title with the first one
    for title in title_entities[1:]:
        similarity = textdistance.jaro_winkler.normalized_similarity(
            title['entity_text'].lower(), filtered_titles[0]['entity_text'].lower()
        )
        
        if similarity < 0.95:
            filtered_titles.append(title)
    
    return filtered_titles


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
    
    # collect results using custom recognizer and filter similar titles
    for text in tqdm(dataset['page_text']):
        
        entities = custom_recognizer(text, model, tokenizer)
        
        # filter for TITLE entities only
        title_entities = [e for e in entities if e['class'] == 'TITLE']
        
        filtered_titles = filter_similar_titles(title_entities)
        
        # take first title if exists, otherwise empty string
        if filtered_titles:
            # Select the title with the longest 'entity_text'
            title = max(filtered_titles, key=lambda x: len(x['entity_text']))['entity_text']
        else:
            title = ""
        
        results.append(title)
    
    # update original JSON with new data
    for item, title in zip(data, results):
        item['patent_title'] = title
    
    # save 
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"updated patent titles in: {os.path.basename(file_path)}")


def process_json_files(folder_path, model, tokenizer):
    
    for filename in os.listdir(folder_path):
        
        if filename.endswith('.json'):
            
            file_path = os.path.join(folder_path, filename)
            
            process_json_file(file_path, model, tokenizer)


model, tokenizer = load_model()
process_json_files(folder_path, model, tokenizer)
######## patent data post-processing entities and relation extraction with GPT4o ###########
##
##
## The code requires a folder structure like the following:
##
## current_working_dir/
## └── input_folder/
##     ├── 1617.json
##     ├── ...
##     ├── 1898.json
##     └── 1899.json
##
## Each JSON file should contain patent data in the following format:
## [
##   {
##     "patent_id": "...",
##     "full_text": [
##       {"page_num": 1, "page_text": "..."},
##       {"page_num": 2, "page_text": "..."},
##       ...
##     ],
##     "front_page_entities": [
##       {
##         "class": "ENTITY_TYPE",
##         "entity_text": "Extracted entity text",
##         "start": start_index,
##         "end": end_index
##       },
##       ...
##     ],
##     ... (other fields)
##   },
##   ... (more patents)
## ]
##
## The script performs the following steps:
## 1. Assigns person IDs for relation extraction
## 2. Uses a mixture of rules and GPT4o to do relation extraction (group OCC, ADD, FIRM with people)
##

import os
import json
import re
import time
import openai
import pandas as pd
from tqdm import tqdm


openai.api_key = ...  # replace with actual API key

input_folder_path = 'gbpatentdata_OCR_output'  # input folder path 



def get_sorted_json_files(folder_path):
    
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    json_files.sort()  # in ascending order

    return json_files




def process_patents_in_file(input_file_path, output_file_path):

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        patents = json.load(infile)

    # read front_page_entities from entities.json, create a lookup dict by patent_id
    with open('entities.json', 'r', encoding='utf-8') as ef:
        entity_entries = json.load(ef)
    entity_map = { ent['patent_id']: ent for ent in entity_entries }

    year = os.path.basename(input_file_path).split('.')[0]
    print(f"\nProcessing year: {year}")

    num_cases = 0
    processed_count = 0

    for patent in tqdm(patents, desc=f"Year {year}", unit="patent"):

        # skip if 'front_page_entities' is not present
        # find matching front_page_entities from entities.json
        if patent['patent_id'] not in entity_map:
            continue
        if 'front_page_entities' not in entity_map[patent['patent_id']] or not entity_map[patent['patent_id']]['front_page_entities']:
            continue

        # skip if patent has already been processed
        if 'front_page_entities_gpt4o' in patent and patent['front_page_entities_gpt4o']:
            continue

        entities = entity_map[patent['patent_id']]['front_page_entities']

        full_text = patent.get('full_text', [])

        if not isinstance(entities, list) or not full_text:
            continue

        # get the lowest page number text
        lowest_page = min(full_text, key=lambda x: x['page_num'])

        page_text = lowest_page['page_text']

        persons = [e for e in entities if e['class'] == 'PER']

        occupations = [e for e in entities if e['class'] == 'OCC']

        addresses = [e for e in entities if e['class'] == 'ADD']

        firms = [e for e in entities if e['class'] == 'FIRM']

        if len(persons) >= 2 and (occupations or addresses or firms):

            num_cases += 1

            entities_str = json.dumps(entities)  # convert list[dict] to JSON string

            system_message = "You are an expert in inferring relations between named entities in historical patent text."

            user_message = f"""
You are an expert in inferring relations between named entities in historical patent text.

You are to output **only** the updated list of dictionaries in JSON format, without any additional text or explanations.

Here is a list of people and entities extracted from a patent document.

Based on the context of the document, assign a new dictionary key 'person_id' to every 'OCC', 'ADD', and 'FIRM' entity.

For 'OCC', 'ADD', and 'FIRM' entities, if they apply to only one person, assign a single integer (e.g., "person_id": 2).

If they apply to multiple people, assign a list of person_ids (e.g., "person_id": [1, 2]).

**Do not include any explanations, comments, or extra text. Return only the updated list of dictionaries in valid JSON format.**

People:
{[p['entity_text'] for p in persons]}

Entities:
{entities_str}

Context:
{page_text}
"""
            max_retries = 5
            retry_count = 0
            backoff_factor = 1
            delay_seconds = 1

            while retry_count < max_retries:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ],
                        max_tokens=16384
                    )

                    assigned_entities = response['choices'][0]['message']['content'].strip()

                    # be nice to the API
                    time.sleep(delay_seconds)

                    if assigned_entities:

                        # extract JSON content from the response
                        json_match = re.search(r'(\{.*\}|\[.*\])', assigned_entities, re.DOTALL)

                        if json_match:
                            json_content = json_match.group(0)
                            try:
                                parsed_entities = json.loads(json_content)
                                # store GPT4o results in the same dictionary from entities.json
                                entity_map[patent['patent_id']]['front_page_entities_gpt4o'] = parsed_entities

                            except json.JSONDecodeError as e:
                                print(f"Error parsing JSON for patent {patent['patent_id']}: {str(e)}")
                                with open('api_response_errors.log', 'a', encoding='utf-8') as log_file:
                                    log_file.write(f"Patent {patent['patent_id']} API response:\n{assigned_entities}\n\n")
                        else:
                            print(f"No JSON content found in API response for patent {patent['patent_id']}")
                            with open('api_response_errors.log', 'a', encoding='utf-8') as log_file:
                                log_file.write(f"Patent {patent['patent_id']} API response:\n{assigned_entities}\n\n")
                    else:
                        print(f"Empty response for patent {patent['patent_id']}.")

                    break  # exit retry loop if successful

                except openai.error.RateLimitError:
                    retry_count += 1
                    wait_time = backoff_factor ** retry_count
                    print(f"Rate limit error. Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    if retry_count == max_retries:
                        print(f"Max retries reached for patent {patent['patent_id']}.")
                        break

                except openai.error.OpenAIError as e:
                    print(f"OpenAI API error for patent {patent['patent_id']}: {str(e)}")
                    break  # do not retry for other OpenAI errors

        processed_count += 1

    # update entities.json with the new GPT4o data
    # by writing out the entire updated entity_map as a list
    updated_entities = list(entity_map.values())
    with open('entities.json', 'w', encoding='utf-8') as ef:
        json.dump(updated_entities, ef, indent=2)


    print(f"Year {year}: Number of patents processed by the API: {num_cases}")
    print(f"Year {year}: Total patents processed: {processed_count}")



def main():
    
    os.makedirs(output_folder_path, exist_ok=True)

    json_files = get_sorted_json_files(input_folder_path)

    for json_file in json_files:
        input_file_path = os.path.join(input_folder_path, json_file)
        output_file_path = os.path.join(output_folder_path, json_file)

        print(f"Processing file: {json_file}")

        process_patents_in_file(input_file_path, output_file_path)



main()
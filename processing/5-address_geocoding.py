import json
import openai
from typing import List, Dict, Set
import jsonlines
import os
from tqdm import tqdm



def load_processed_patents(log_file: str) -> Set[str]:
    """Load the set of already processed patent IDs."""
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return set(line.strip() for line in f)
    return set()



def load_existing_output(output_path: str) -> Dict[str, dict]:
    """Load existing processed patents from output file."""
    existing_data = {}
    if os.path.exists(output_path):
        with jsonlines.open(output_path) as reader:
            for patent in reader:
                if 'patent_id' in patent:
                    existing_data[patent['patent_id']] = patent
    return existing_data



def mark_patent_processed(log_file: str, patent_id: str):
    """Mark a patent as processed by appending to log file."""
    with open(log_file, 'a') as f:
        f.write(f"{patent_id}\n")



def extract_addresses_with_context(patent_data: Dict) -> List[Dict]:
    addresses = []
    for idx, entity in enumerate(patent_data['front_page_entities']):
        if entity['class'] == 'ADD':
            addresses.append({
                'idx': idx,
                'text': entity['entity_text'],
                'person_id': entity['person_id']
            })
    return addresses



def create_gpt_prompt(addresses: List[Dict]) -> str:
    prompt = """These addresses are from British historical patent records, but the locations mentioned could be anywhere in the world, not just the UK. Please provide latitude and longitude coordinates for each address.

Important guidelines:
1. The addresses could be from any country, not just the UK
2. For ambiguous place names (e.g., "Rochester" which exists in UK and US), use the UK location only if there's no clear indication of another country
3. Return null for both latitude and longitude if:
   - The location is too vague
   - There are multiple possible locations and no clear context
   - The historical location cannot be accurately mapped to modern coordinates

Return results as a JSON array with 'text', 'latitude', 'longitude'.

Addresses to process:
"""
    for addr in addresses:
        prompt += f"- {addr['text']}\n"
    return prompt



def parse_gpt_response(response_text: str) -> List[Dict]:
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return []



def count_total_patents(input_path: str) -> int:
    """Count total number of patents in the input file."""
    count = 0
    with jsonlines.open(input_path) as reader:
        for _ in reader:
            count += 1
    return count



def process_file(input_path: str, output_path: str, api_key: str, processed_log: str):
    openai.api_key = api_key
    processed_patents = load_processed_patents(processed_log)
    existing_output = load_existing_output(output_path)
    
    # Backup existing output file if it exists
    if os.path.exists(output_path):
        backup_path = f"{output_path}.backup"
        os.rename(output_path, backup_path)
        print(f"Created backup of existing output at {backup_path}")
    
    # Get total count for progress bar
    total_patents = count_total_patents(input_path)
    
    # Process from input file
    with jsonlines.open(input_path) as reader, \
         jsonlines.open(output_path, mode='w') as writer:
        
        for patent in tqdm(reader, total=total_patents, desc="Processing patents"):
            patent_id = patent['patent_id']
            
            # If we've already processed this patent, use existing data
            if patent_id in existing_output:
                writer.write(existing_output[patent_id])
                continue
            
            # Skip if marked as processed but not in existing output
            # (this shouldn't happen often, but handles edge cases)
            if patent_id in processed_patents:
                writer.write(patent)
                continue
                
            addresses = extract_addresses_with_context(patent)
            
            if not addresses:
                writer.write(patent)
                mark_patent_processed(processed_log, patent_id)
                continue
                
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides precise geographic coordinates for historical addresses from patent records. Consider international locations while defaulting to UK only for ambiguous cases with no clear context."},
                        {"role": "user", "content": create_gpt_prompt(addresses)}
                    ],
                    temperature=0.0
                )
                
                coordinates = parse_gpt_response(response.choices[0].message.content)
                
                for addr, coord in zip(addresses, coordinates):
                    idx = addr['idx']
                    patent['front_page_entities'][idx]['latitude'] = coord.get('latitude')
                    patent['front_page_entities'][idx]['longitude'] = coord.get('longitude')
                
            except Exception as e:
                tqdm.write(f"Error processing patent {patent_id}: {str(e)}")
            
            writer.write(patent)
            mark_patent_processed(processed_log, patent_id)
            tqdm.write(f"Processed {patent_id}")




INPUT_FILE = "../300YearsOfBritishPatents/entities.jsonl"
OUTPUT_FILE = "../300YearsOfBritishPatents/entities_address_modif_consolidated.jsonl"
PROCESSED_LOG = "../300YearsOfBritishPatents/processed_patents.txt"
API_KEY = ...
    
process_file(INPUT_FILE, OUTPUT_FILE, API_KEY, PROCESSED_LOG)
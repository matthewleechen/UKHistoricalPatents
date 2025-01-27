import json
from collections import defaultdict

def process_jsonl(input_file, output_file):
    # Counter for removed PER dictionaries
    removed_count = 0
    # Counter for duplicate patent-inventor pairs
    duplicate_pairs = 0
    
    # Process file line by line to handle large files efficiently
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            patent = json.loads(line)
            
            # Group PER entities by inventor_id and sort by start index
            per_entities = defaultdict(list)
            for i, entity in enumerate(patent['front_page_entities']):
                if entity['class'] == 'PER':
                    per_entities[entity['inventor_id']].append((i, entity))
            
            # Track indices to remove and person_id mapping
            indices_to_remove = set()
            person_id_mapping = {}
            
            # Find duplicates and create mapping
            for inventor_id, entities in per_entities.items():
                if len(entities) > 1:
                    # Sort by start index
                    sorted_entities = sorted(entities, key=lambda x: x[1]['start'])
                    # Keep first occurrence, mark others for removal
                    kept_person_id = sorted_entities[0][1]['person_id'][0]
                    for idx, entity in sorted_entities[1:]:
                        indices_to_remove.add(idx)
                        person_id_mapping[entity['person_id'][0]] = kept_person_id
                    duplicate_pairs += len(entities) - 1
            
            # Create new front_page_entities list
            new_entities = []
            for i, entity in enumerate(patent['front_page_entities']):
                if i not in indices_to_remove:
                    # Update person_id references if needed
                    if entity['class'] in ['ADD', 'OCC', 'FIRM']:
                        new_person_id = [
                            person_id_mapping.get(pid, pid) 
                            for pid in entity['person_id']
                            if pid not in person_id_mapping.keys()
                        ]
                        if new_person_id:  # Only keep if there are valid references
                            entity['person_id'] = new_person_id
                            new_entities.append(entity)
                    else:
                        new_entities.append(entity)
            
            removed_count += len(indices_to_remove)
            
            # Update patent with new entities
            patent['front_page_entities'] = new_entities
            
            # Write to output file
            json.dump(patent, fout)
            fout.write('\n')
    
    return removed_count, duplicate_pairs
    

input_file = '../300YearsOfBritishPatents_decompressed/entities_address_modif_consolidated.jsonl'
output_file = '../300YearsOfBritishPatents_decompressed/entities_address_modif_consolidated2.jsonl'

removed_count, duplicate_pairs = process_jsonl(input_file, output_file)
print(f"Number of PER dictionaries removed: {removed_count}")
print(f"Number of duplicate patent-inventor pairs: {duplicate_pairs}")
print(f"Verification: Numbers match: {removed_count == duplicate_pairs}")
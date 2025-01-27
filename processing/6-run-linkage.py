import os
import json
import pandas as pd
import numpy as np
import linktransformer as lt


def build_patent_inventor_dataframe(entities_json_path, ocr_folder_path):
    """
    1. Reads all entries from entities.json
    2. Builds a lookup { patent_id -> patent_title } from gbpatentdata_OCR_output
    3. Creates a DataFrame of individuals (PER) and their associated OCC, ADD, FIRM
       plus the patent_title, year, and crucially the person_id for round-trip.
    """

    # 1) Read all entity entries from entities.json
    with open(entities_json_path, 'r', encoding='utf-8') as f:
        entity_data = json.load(f)

    # 2) Build a lookup for { patent_id -> patent_title } from gbpatentdata_OCR_output
    title_map = {}
    for filename in os.listdir(ocr_folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(ocr_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as infile:
                patent_list = json.load(infile)
                for p in patent_list:
                    # If "patent_title" is stored differently, adjust accordingly
                    pid = p.get('patent_id')
                    if pid and 'patent_title' in p:
                        title_map[pid] = p['patent_title']

    # 3) Accumulate records
    all_records = []
    for entry in entity_data:
        patent_id = entry.get('patent_id')
        year = entry.get('year')
        entities = entry.get('front_page_entities', [])  # Adjust key if needed

        # Build dictionary [person_id -> row] just like the original approach
        person_data = {}

        # 3A) Collect PER entities
        for e in entities:
            if e.get("class") == "PER":
                # person_id is a list
                pids = e["person_id"]
                for pid in pids:
                    if pid not in person_data:
                        person_data[pid] = {
                            "patent_id": patent_id,
                            "person_id": pid,        # <--- store person_id
                            "year": year,
                            "name": e["entity_text"],
                            "occupation": None,
                            "address": None,
                            "firm": None,
                            "patent_title": title_map.get(patent_id, None),
                            # We'll fill in inventor_id later via linktransformer
                        }

        # 3B) Map OCC/ADD/FIRM to the correct person(s)
        for e in entities:
            if e["class"] in ["OCC", "ADD", "FIRM"]:
                pids = e["person_id"] if isinstance(e["person_id"], list) else [e["person_id"]]
                for pid in pids:
                    if pid in person_data:
                        if e["class"] == "OCC" and person_data[pid]["occupation"] is None:
                            person_data[pid]["occupation"] = e["entity_text"]
                        elif e["class"] == "ADD" and person_data[pid]["address"] is None:
                            person_data[pid]["address"] = e["entity_text"]
                        elif e["class"] == "FIRM" and person_data[pid]["firm"] is None:
                            person_data[pid]["firm"] = e["entity_text"]

        # Add to the big list
        all_records.extend(person_data.values())

    # 4) Return a DataFrame of all matched records
    return pd.DataFrame(all_records)


def assign_inventor_ids_and_save(df, entities_json_path):
    """
    1. Clusters df rows via linktransformer
    2. Produces a { (patent_id, person_id) -> inventor_id } map
    3. Reopens entities.json, inserts inventor_id into each PER entity
       matching the same (patent_id, person_id).
    4. Overwrites entities.json with the updated data.
    """

    # 1) Perform linktransformer clustering
    df_lm_matched = lt.cluster_rows(
        df,
        model='gbpatentdata/lt-patent-inventor-linking',
        on=['name', 'occupation', 'year', 'address', 'firm', 'patent_title'],
        cluster_type='SLINK',
        cluster_params={
            'threshold': 0.1,
            'min cluster size': 1,
            'metric': 'cosine'
        }
    )

    # Suppose the returned DataFrame has a column like 'cluster_id'.
    # We'll rename it to inventor_id for clarity:
    df_lm_matched.rename(columns={'cluster': 'inventor_id'}, inplace=True)

    # 2) Create a map: (patent_id, person_id) -> inventor_id
    cluster_map = {}
    for idx, row in df_lm_matched.iterrows():
        key = (row['patent_id'], row['person_id'])
        cluster_map[key] = row['inventor_id']

    # 3) Load existing entities.json
    with open(entities_json_path, 'r', encoding='utf-8') as ef:
        entity_data = json.load(ef)

    # For each entry in entities.json, update PER with inventor_id if found
    for entry in entity_data:
        patent_id = entry.get('patent_id')
        entities_list = entry.get('front_page_entities', [])

        for e in entities_list:
            if e.get('class') == 'PER':
                # e["person_id"] is a list
                pids = e["person_id"]
                for pid in pids:
                    # Check if we have a match in cluster_map
                    key = (patent_id, pid)
                    if key in cluster_map:
                        e['inventor_id'] = cluster_map[key]

    # 4) Save the updated JSON back
    with open(entities_json_path, 'w', encoding='utf-8') as ef:
        json.dump(entity_data, ef, indent=2)

    return df_lm_matched  # if you still want the final DataFrame for inspection


# 1) Build patent-inventor df
df = build_patent_inventor_dataframe(
    entities_json_path='entities.json',
    ocr_folder_path='gbpatentdata_OCR_output'
)

# 2) Cluster and save inventor_id back to entities.json
df_matched = assign_inventor_ids_and_save(df, 'entities.json')

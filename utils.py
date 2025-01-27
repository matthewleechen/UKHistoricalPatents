from datasets import load_dataset
import pandas as pd
import re


def load_base_datasets():
    """Load the core patent datasets."""
    dataset_all_years = load_dataset(
        "matthewleechen/300YearsOfBritishPatents",
        data_files="texts.jsonl.gz"
    )
    
    dataset_all_entities = load_dataset(
        "matthewleechen/300YearsOfBritishPatents",
        data_files="entities.jsonl.gz"
    )
    
    return dataset_all_years, dataset_all_entities




def load_kpst_scores():
    """Load all variants of the KPST breakthrough scores."""
    variants = {
        'fh1': "breakthrough_scores_fh1_bh5_maxdf100.csv",
        'fh5': "breakthrough_scores_fh5_bh5_maxdf100.csv",
        'fh10': "breakthrough_scores_fh10_bh5_maxdf100.csv",
        'fh20': "breakthrough_scores_fh20_bh5_maxdf100.csv"
    }
    
    scores = {}
    for variant, filename in variants.items():
        scores[variant] = load_dataset(
            "matthewleechen/300YearsOfBritishPatents_KPST",
            data_files=filename
        )
    
    return scores




def process_patent_inventors(dataset_entities):
    """
    Process HuggingFace entities dataset into a patent-inventor dataframe.
    
    Args:
        dataset_entities: HuggingFace dataset from entities.jsonl.gz
        
    Returns:
        pd.DataFrame with columns:
            - patent_id: string
            - inventor_id: int
            - year: int
            - name: string
            - occupation: string
            - address: string
            - firm: string
    """
    all_records = []
    
    for row in dataset_entities["train"]:
        patent_id = row["patent_id"]
        year = row["year"]
        person_data = {}

        # First get person entities
        for entity in row["front_page_entities"]:
            if entity["class"] == "PER":
                for pid in entity["person_id"]:
                    if pid not in person_data:
                        person_data[pid] = {
                            "patent_id": patent_id,
                            "inventor_id": entity.get("inventor_id"),
                            "year": year,
                            "name": entity["entity_text"],
                            "occupation": None,
                            "address": None,
                            "firm": None,
                        }

        # Then map metadata to persons
        for entity in row["front_page_entities"]:
            if entity["class"] in ["OCC", "ADD", "FIRM"]:
                for pid in entity["person_id"]:
                    if pid in person_data:
                        if entity["class"] == "OCC" and person_data[pid]["occupation"] is None:
                            person_data[pid]["occupation"] = entity["entity_text"]
                        elif entity["class"] == "ADD" and person_data[pid]["address"] is None:
                            person_data[pid]["address"] = entity["entity_text"]
                        elif entity["class"] == "FIRM" and person_data[pid]["firm"] is None:
                            person_data[pid]["firm"] = entity["entity_text"]

        all_records.extend(list(person_data.values()))

    return pd.DataFrame(all_records)




def standardize_occupation(occupation):
    
    """Helper function to standardize occupation strings."""
    
    if pd.isna(occupation):
        return occupation

    occupation = str(occupation).lower()
    occupation = re.sub(r'[^\w\s]', ' ', occupation)
    occupation = ' '.join(occupation.split())

    if occupation in ['gentleman', 'gentlemen', 'gent', 'gentle']:
        return 'gentleman'
        
    if occupation in ['esquire', 'esquires', 'esq', 'esquier']:
        return 'esquire'

    words = occupation.split()
    if len(words) > 0:
        last_word = words[-1]
        if last_word.endswith('s') and not (
            last_word.endswith('ss') or
            last_word.endswith('ics') or
            last_word.endswith('ies') or
            last_word in ['gas', 'glass', 'brass']
        ):
            words[-1] = last_word[:-1]

    return ' '.join(words)

######## preprocess patent texts ###########                    
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
## the code outputs a "word_tokens" key to each patent dictionary, containing
## a list of preprocessed tokens
##
## Preprocessing steps: 
##
## 1) Dissolve html tags, non-ASCII patterns
##
## 2) Replace hyphens with spaces 
##
## 3) Tokenize text with en_core_web_lg spaCy model, removing numbers and punctuation  
##
##
##

import spacy
import re
import os
import orjson 
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path


nlp = spacy.load('en_core_web_lg')

nlp.max_length = 100000000


html_tag_pattern = re.compile(r'<.*?>')

non_ascii_pattern = re.compile(r'[^\x00-\x7F]+')



def clean_text(text: str) -> str:
    """
    Cleans the input text by removing HTML tags, non-ASCII characters, 
    and hyphenations.
    
    Args:
        text (str): The text to be cleaned.
    
    Returns:
        str: The cleaned text.
    """
    # remove HTML tags
    text = html_tag_pattern.sub('', text)
    
    # remove non-ASCII characters
    text = non_ascii_pattern.sub('', text)
    
    # handle hyphenations (replace hyphens with spaces)
    text = text.replace('-', ' ')
    
    return text



def preprocess_text(text: str) -> list:
    """
    Preprocess the input text using spaCy, keeping stopwords but removing
    punctuation and numerical tokens.
    
    Args:
        text (str): The text to be preprocessed.
    
    Returns:
        list: List of cleaned, tokenized words.
    """
    # clean text
    text = clean_text(text)

    # run preprocessing with spaCy
    doc = nlp(text)

    # keep stopwords but filter out punctuation and numbers
    tokens = [
        # lower case
        token.text.lower() for token in doc 
        if (
            not token.is_punct 
            and not token.is_space 
            and not token.like_num
        )
    ]
    
    return tokens



def preprocess_patent(patent: dict) -> dict:
    """
    Preprocesses a patent by concatenating its page-level texts 
    and tokenizing the result.
    
    Args:
        patent (dict): Patent dictionary with 'full_text' and other fields.
    
    Returns:
        dict: Patent dictionary with added 'word_tokens'.
    """
    # concatenate page-level texts
    full_text = " ".join(
        page["page_text"] for page in sorted(patent["full_text"], key=lambda x: x["page_num"])
    )

    tokens = preprocess_text(full_text)
    patent['word_tokens'] = tokens
    
    return patent



def preprocess_file(file_path: Path):
    """
    Processes a JSON file containing patents, cleans and tokenizes the text, 
    and saves the updated patents to the same file.
    
    Args:
        file_path (Path): Path to the file to be processed.
    """
    with open(file_path, 'rb') as f:
        patents = orjson.loads(f.read())

    with mp.Pool(processes=mp.cpu_count()) as pool:
        # process patents in parallel
        updated_patents = list(
            tqdm(
                pool.imap(preprocess_patent, patents),
                total=len(patents),
                desc=f"processing {os.path.basename(file_path)}..."
            )
        )

    with open(file_path, 'wb') as f:
        f.write(orjson.dumps(updated_patents))
        
    return patents
    

input_folder = Path('gbpatentdata_OCR_output')

json_files = list(input_folder.glob('*.json'))

    
for file_path in tqdm(json_files, desc="Processing files"):
    preprocess_file(file_path)
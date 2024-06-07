import json
import os
import numpy as np
import pandas as pd

OPENAI_KEY = ''

def get_openai_key():
    return OPENAI_KEY

def parse_jsonl(file):
    data_list = [json.loads(line) for line in file.split('\n') if line.strip()]
    return data_list


def save_jsonl(json_list, file_path):
    with open(file_path, 'w') as file:
        for b in json_list:
            json_b = json.dumps(b)
            file.write(json_b + '\n')
            
def read_jsonl(file_path):
    jsonl = []
    with open(file_path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        jsonl.append(result)
    return jsonl

def convert_ndarrays(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
def get_datashop_transaction(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if 'tx_All_Data' in file and file.endswith('.txt'):
                return pd.read_csv(os.path.join(root, file), delimiter='\t', low_memory=False)
            
    raise ValueError(f'All Transaction Data is not observed on the desired path: {directory_path}')
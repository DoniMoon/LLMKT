# This file generates test dataframe of silhouette_scores experiment.
# do cluster for the best silhouette. 
# Used for generating figure 3 of the paper.

import os
import json
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from utils import *
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
import argparse
from pyafm.roll_up import transaction_to_student_step


def most_frequent_element(lst):
    if not lst:  
        return None, 0
    counter = Counter(lst)
    most_common_element, count = counter.most_common(1)[0]
    return most_common_element, count



content_path = Path('./')
dset_config = json.load(open(os.path.join(content_path,'config.json')))

def find_local_max_indices(array, bins):
    offset = 10
    array = np.array(array[offset:])
    length = len(array)
    bin_size = length // bins
    indices = []


    for i in range(bins):
        start = i * bin_size
        if i == bins - 1:
            end = length
        else:
            end = start + bin_size
        segment = array[start:end]
        
        local_max_index = np.argmax(segment)
        indices.append(offset + start + local_max_index)
    
    return indices

def find_max_indices(array, offset = 10):
    return offset + np.argmax(array[offset:])



def add_local_max_clusters(dset, model_name, bins = 10):
    processed = json.load(open(content_path / 'resources'/ dset/ f'{model_name}_processed_embedings.json'))
    print(dset)
    descriptions = []
    embeddings = []
    names = []
    for i in processed:
        for kc in i['kcs']:
            descriptions.append(kc['description'])
            embeddings.append(kc['embedding'])
            names.append(kc['name'])
    scores = json.load(open(content_path / 'resources'/ dset/ f'{model_name}_cluster_scores.json'))
    cluster_nums = find_local_max_indices(scores['silhouette'], bins)
    max_score_idx = find_max_indices(scores['silhouette'])
    print(max_score_idx)    
    dshop_df = pd.read_csv(os.path.join(content_path,'resources', dset, 'openai_3_datashop_form.txt'), sep='\t', low_memory=False)
    jsonl = read_jsonl(os.path.join(content_path,'resources', dset, 'merged_gpt_results.jsonl'))
    content_data = json.load(open(os.path.join(content_path,'resources', dset, 'openai_3_content_data.json')))

    
    for cluster_num in cluster_nums:
        kmeans = KMeans(n_clusters=cluster_num, random_state=42)
        kmeans.fit(embeddings)
        clusters = kmeans.labels_
        embeddings = np.array(embeddings)
        
        idx2cluster = {}
        for i in range(cluster_num):
            cluster_indices = np.where(clusters == i)[0]
            for idx in cluster_indices:
                idx2cluster[idx] = i
    
        batch_id2kcs= {}
        for j, p in zip(jsonl, processed):
            batch_id = j['custom_id']
            batch_id2kcs[batch_id] = set([idx2cluster[k['id']] for k in p['kcs']])
        
        content_step_name2ctags = {}
        for c in content_data:
            if 'db_step_name' not in c:
                continue
            _key = (c['problem_id'],c['db_step_name'])
            content_step_name2ctags[_key] = batch_id2kcs[c['batch_id']]
            
        def process(x):
            c = content_step_name2ctags.get((x['Problem Name'],x['Step Name']))
            return '~~'.join([ str(i) for i in c])

        dshop_df[f'KC (GPT_cluster_{cluster_num})'] = dshop_df.apply(process, axis=1)
        if cluster_num == max_score_idx:
            dshop_df['KC (Max_score)'] = dshop_df.apply(process, axis=1)

    dshop_df.to_csv(os.path.join(content_path,'resources', dset, f'{model_name}_datashop_form.txt'), sep='\t', index=False)
    transaction_to_student_step(open(os.path.join(content_path,'resources', dset, f'{model_name}_datashop_form.txt'),'r'))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get dataset name')
    dataset_choices = list(dset_config.keys())
    parser.add_argument(
        'dataset', 
        type=str, 
        choices=dataset_choices + ['all'],
        default='oli_statics',
        nargs='?',
        help='The dataset string to be processed. Choices: ' + ', '.join(dataset_choices)
    )
    parser.add_argument(
        'model', 
        type=str, 
        choices=['t5', 'openai_3'],
        default='openai_3',
        nargs='?',
        help='select embedding model. t5 or openai_3 '
    )
    args = parser.parse_args()
    target_dsets = [args.dataset] if args.dataset != 'all' else dataset_choices
    for dset in target_dsets:
        add_local_max_clusters(dset,args.model)
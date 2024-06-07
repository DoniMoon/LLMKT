from datetime import datetime
from tqdm import tqdm
import os
import json
from pathlib import Path
import argparse
from collections import OrderedDict
import pandas as pd
import random
import numpy as np
from scipy import sparse

content_path = Path('./')

def convert_to_timestamp(time_str):
    dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    timestamp = dt.timestamp()
    return int(timestamp)

def skill2unique_str(skill_str):
    return '~~'.join(sorted([s for s in skill_str.split('~~')]))


class InputStack:
    def __init__(self):
        self.stack = OrderedDict()

    def add_input(self, new_input):
        if new_input in self.stack:
            return list(self.stack.keys()).index(new_input)
        else:
            self.stack[new_input] = None
            return len(self.stack) - 1
        
def nancheck(x):
    if type(x) != str and np.isnan(x):
        return ""
    return x

def get_max_skill_num(user_data, skill_name):
    
    skill_nums = []
    for _, x in user_data.iterrows():
        skill_nums.extend(list(map(int, nancheck(x[f"KC ({skill_name})"]).split('~~'))))
    return max(skill_nums)

def get_test_item_ids(Q_mat, train_test_ratio):
    # if train_test_ratio = 5, train_items : test_items = 5:1
    budget = 10000
    random.seed(42)
    for i in range(budget):
        test_ids = random.sample(list(range(Q_mat.shape[0])), k=int(Q_mat.shape[0]/(train_test_ratio+1)))
        train_skill_cnt, test_skill_cnt = np.zeros(Q_mat.shape[1]),np.zeros(Q_mat.shape[1])
        for item_id in range(Q_mat.shape[0]):
            if item_id in test_ids:
                test_skill_cnt += Q_mat[item_id]
            else:
                train_skill_cnt += Q_mat[item_id]
        if all(train_skill_cnt) and all(test_skill_cnt):
            return test_ids
    
    raise RuntimeError('No suitable test split found')
    

def generate_kt_df(dset, skill_name, is_random=False, is_cold_start=False):
    kt_data = []
    invalid_cnt = 0
    item_stack = InputStack()
    integrated_skill_stack = InputStack()
    skill_stack = InputStack()
    dshop_df = pd.read_csv(os.path.join(content_path,'resources', dset, 'openai_3_datashop_form-rollup.txt'), sep='\t', low_memory=False)
    Anon_Id2user_id = {}

    for i,aid in enumerate(set(dshop_df['Anon Student Id'])):
        Anon_Id2user_id[aid] = i

    item2kcs=dict()
    max_skill_num = get_max_skill_num(dshop_df, skill_name) if is_random else None
    
    for aid in tqdm(list(Anon_Id2user_id.keys())):
        user_data = dshop_df[dshop_df['Anon Student Id'] == aid]
        timestamp_mapper = {}
        user_min_time = min(list(set(convert_to_timestamp(i) for i in user_data['Step End Time'])))
        for i, x in user_data.iterrows():
            item_id = item_stack.add_input((x['Problem Name'],x['Step Name']))            
            skill_str = nancheck(x[f"KC ({skill_name})"]) if not is_random else str(random.randint(1, max_skill_num))
            item2kcs[item_id] = [skill_stack.add_input(s) for s in skill_str.split('~~')]
            kt_data.append({
                'user_id': Anon_Id2user_id[aid],
                'item_id': item_id,
                'timestamp': convert_to_timestamp(x['Step End Time']) - user_min_time,
                'correct': 1 if x['First Attempt'] == 'correct' else 0,
                'skill_id': integrated_skill_stack.add_input(skill2unique_str(skill_str))
            })
            


    Q_mat = np.zeros((len(list(item_stack.stack.keys())), len(list(skill_stack.stack.keys()))))    
    for item_id, kcs in item2kcs.items():
        for kc in kcs:
            Q_mat[item_id, kc] = 1

    l = pd.DataFrame.from_dict(kt_data)    
    skill_dir_name = skill_name.replace("'",'').replace(' ','')
    llmkt_path = os.path.join(content_path, '../kt_benchmark/data', f"{dset}_{skill_dir_name}")
    if not os.path.exists(llmkt_path):
        os.mkdir(llmkt_path)
    sparse.save_npz(os.path.join(llmkt_path, "q_mat.npz"), sparse.csr_matrix(Q_mat))
    l.to_csv(os.path.join(llmkt_path, 'preprocessed_data.csv'),sep='\t')

    
    random.seed(42)
    user_ids = list(set(l['user_id']))
    test_user_ids = random.sample(user_ids, k=int(len(user_ids)/6))
    i = l.apply(lambda x: x['user_id'] in test_user_ids, axis=1)
    
    if is_cold_start:
        test_item_ids = get_test_item_ids(Q_mat, train_test_ratio=5)
        j = l.apply(lambda x: x['item_id'] in test_item_ids, axis=1)
        l[(i==False) & (j==False)].to_csv(os.path.join(llmkt_path, 'preprocessed_data_train.csv'),sep='\t')
        l[(i) & (j)].to_csv(os.path.join(llmkt_path, 'preprocessed_data_test.csv'),sep='\t')
    else:
        l[i==False].to_csv(os.path.join(llmkt_path, 'preprocessed_data_train.csv'),sep='\t')
        l[i].to_csv(os.path.join(llmkt_path, 'preprocessed_data_test.csv'),sep='\t')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get dataset name')
    dset_config = json.load(open(os.path.join(content_path,'config.json')))
    dataset_choices = list(dset_config.keys())
    parser.add_argument(
        'dataset', 
        type=str, 
        choices=dataset_choices + ['all'],
        default='all',
        nargs='?',
        help='The dataset string to be processed. Choices: ' + ', '.join(dataset_choices)
    )
    parser.add_argument(
        'skill_name', 
        type=str, 
        default="Ours",
        nargs='?',
        help='select skill name'
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Flag to indicate if random tag selection should be used'
    )
    parser.add_argument(
        '--item_cold_start',
        action='store_true',
        help='Flag to indicate item cold start setting'
    )
    args = parser.parse_args()
    target_dsets = [args.dataset] if args.dataset != 'all' else dataset_choices
    for dset in target_dsets:
        generate_kt_df(dset, args.skill_name, args.random, args.item_cold_start)
from datetime import datetime
from tqdm import tqdm
import os
import json
from pathlib import Path
import argparse
from collections import OrderedDict, Counter, defaultdict
import pandas as pd
import random
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from utils import *
from openai_batch import *
from pyafm.roll_up import transaction_to_student_step

content_path = Path('./')


def get_part_input(parsed_data, content_id, qid, step_id):
    for p in parsed_data:
        if p['problem'] == content_id and p['qid']==qid:
            for part in p['parts']:
                if part['step_id'] == step_id:
                    return part['input']


def generate_datashop_txt(dset, model_name, is_single_kc = False):
    transaction_data = get_datashop_transaction(os.path.join(content_path,dset_config[dset]['transaction_path']))
    if is_single_kc:
        model_name = model_name + '_single'
    processed = json.load(open(os.path.join(content_path,'resources', dset, f'{model_name}_processed_embedings.json'))) # order is same with gpt4_complete
    parsed_data = json.load(open(os.path.join(content_path,'resources', dset, 'parsed_steps.json')))
    cnt = 0
    content_data = []
    dup_check = []
    for idx, question_content in enumerate(parsed_data):

        user_conts = question_to_prompt(question_content,question_content['file_path'])
        for step, (step_idx, user_cont ) in zip(question_content['steps'], enumerate(user_conts)):
            step_id = step['step_id']
            content_obj = {
                'item_id': cnt,
                'problem_id': question_content['problem'],
                'question_id': question_content['qid'],
                'step_id': step_id,
                'batch_id': f"request-{idx}-{step_idx}"
            }
            dup_checker = (content_obj['problem_id'], content_obj['step_id'], content_obj['question_id'])
            if dup_checker not in dup_check:
                dup_check.append(dup_checker)
                content_data.append(content_obj)
                cnt += 1

    content2question_step_ids = defaultdict(list)
    for x in content_data:
        content2question_step_ids[x['problem_id']].append((x['question_id'],x['step_id']))

    content2step_names = defaultdict(set)
    for i,x in transaction_data.iterrows():
        content2step_names[x['Problem Name']].add(x['Step Name'])

    for content_id, step_names in content2step_names.items():
        if content_id not in content2question_step_ids:  # 14 problems
            continue
        question_step_ids = content2question_step_ids[content_id]
        # for step_name in step_names:
        #     qid_sid = step_name.split(' ')[0]
        for question_step_id in question_step_ids:
            my_question_id, my_step_id = question_step_id
            candidate_steps = [s for s in step_names if type(s)==str and s.startswith(my_question_id+ '_')]
            if len(candidate_steps) ==0: # only single case, q1 of tutor_20_3_0
                continue
            matched_step = [s for s in candidate_steps if s.split(my_question_id + '_')[1].startswith(my_step_id+ ' ')]
            if len(candidate_steps) == 1:
                matched_step = candidate_steps
            if len(matched_step)!= 1:
                part_input = get_part_input(parsed_data, content_id, my_question_id, my_step_id)
                matched_step = [s for s in candidate_steps if s.startswith(f"{my_question_id}_{part_input} ")]
            if len(matched_step)!= 1:
                continue
                # raise ValueError('Unmatched step left')
            for c in content_data:
                if c['problem_id'] == content_id and c['question_id'] == my_question_id and c['step_id'] == my_step_id:
                    c['db_step_name'] = matched_step[0]

    content_step_name2c = {}
    for c in content_data:
        if 'db_step_name' not in c:
            continue
        _key = (c['problem_id'],c['db_step_name'])
        content_step_name2c[_key] = c

    def process(x):
        c = content_step_name2c.get((x['Problem Name'],x['Step Name']))
        if not c:
            print(x['Problem Name'],x['Step Name'])
            return np.nan
        if 'tags' not in c:
            print('No tags: ',c)
            return np.nan
        return '~~'.join([ str(i) for i in set(c['tags'])])
    jsonl = read_jsonl(os.path.join(content_path,'resources', dset, 'merged_gpt_results.jsonl'))

    batch_id2kcs= {}
    for j, p in zip(jsonl, processed):
        batch_id = j['custom_id']
        batch_id2kcs[batch_id] = [k['kc_id'] for k in p['kcs']]

    final_content_data = []
    for c in content_data:
        if c['batch_id'] in batch_id2kcs:
            c['tags'] = batch_id2kcs[c['batch_id']]
            final_content_data.append(c)

    json.dump(final_content_data, open(os.path.join(content_path,'resources', dset, f'{model_name}_content_data.json'),'w'))
    transaction_data['KC (Ours)'] = transaction_data.apply(process, axis=1)
    print(len(transaction_data))
    new_df = transaction_data.dropna(subset=['KC (Ours)'])
    print(len(new_df))
    new_df.to_csv(os.path.join(content_path,'resources', dset, f'{model_name}_datashop_form.txt'), sep='\t', index=False)
    transaction_to_student_step(open(os.path.join(content_path,'resources', dset, f'{model_name}_datashop_form.txt'),'r'))

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
        'model', 
        type=str, 
        choices=['t5', 'openai_3'],
        default='openai_3',
        nargs='?',
        help='select embedding model. t5 or openai_3 '
    )
    parser.add_argument(
        '--single_kc',
        action='store_true',
        help='Disable multiple KCs'
    )
    args = parser.parse_args()
    target_dsets = [args.dataset] if args.dataset != 'all' else dataset_choices
    for dset in target_dsets:
        generate_datashop_txt(dset, args.model, args.single_kc)
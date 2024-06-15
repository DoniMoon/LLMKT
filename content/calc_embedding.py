import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from utils import *

client = OpenAI(api_key = OPENAI_KEY)
content_path = Path('./')


def get_embedding(dset, model='t5', is_single=False):
    jsonl = read_jsonl(content_path / 'resources' / dset/ 'merged_gpt_results.jsonl')

    print(f"{dset} : total {len(jsonl)} steps.")
    processed = []
    split_jsonl = jsonl[len(processed):]
    
    for i in tqdm(jsonl):
        try:
            res = json.loads(i['response']['body']['choices'][0]['message']['content'])
        except:
            continue
            
        if 'knowledge_components' not in res:
            print('exception_no_kc')
            continue
        
        kcs = []
        raw_kcs = []
        if is_single:
            names = []
            descriptions = []
            for kc in res['knowledge_components']:
                if 'description' not in kc or 'name' not in kc:
                    continue
                names.append(kc['name'])
                descriptions.append(kc['description'])
                
            combined_kc = {
                'name': '~~'.join(names),
                'description': '\n- '.join(['Knowledge required to solve the problem:'] + descriptions)
            }
            raw_kcs = [combined_kc]
        else:
            raw_kcs = res['knowledge_components']
        for kc in raw_kcs:
            if 'description' not in kc or 'name' not in kc:
                print(kc)
                continue
            if model == 't5':
                kc['embedding'] = model.encode(kc['description'])
            elif model == 'openai_3':
                try:
                    r = client.embeddings.create(
                          model="text-embedding-3-large",
                          input=kc['description'],
                          encoding_format="float"
                        )
                    kc['embedding'] = r.data[0].embedding
                except KeyError as e:
                    print(f"Unexpected response from opanai : {e}")
                    continue
            else:
                raise ValueError('Invalid model name')
            kcs.append(kc)
        processed.append({
            'response': res,
            'kcs': kcs
        })
    try:
        single_postfix = '_single' if is_single else ''
        json.dump(convert_ndarrays(processed), open(content_path / 'resources'/ dset/ f'processed_{model}{single_postfix}_embedings.json','w'))
    except:
        return processed
    return processed


if __name__ == '__main__':
    dset_config = json.load(open(os.path.join(content_path,'config.json')))
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
        default='t5',
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
        get_embedding(dset,args.model,args.single_kc)
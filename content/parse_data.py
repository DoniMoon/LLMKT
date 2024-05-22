import os
import json
import itertools

from pathlib import Path
import pandas as pd

from bs4 import BeautifulSoup, Comment
import re

import argparse
from openai_batch import parsed_data2batch_list

def clean_text(text):
    text = re.sub(r'&#160;', ' ',text)
    text = re.sub(r'\n{2,}', '<br>', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'<br>', '\n', text)
    return re.sub(r'\s{2,}', ' ', text)

def extract_questions(data_root, file_path, soup, valid_problems):
    problem_parsed = []
    problems = soup.find_all(class_='problem')
    for problem in problems:
        problem_name = problem.find(class_='problem-name').get_text()
        # if problem_name not in valid_problems:
        #     continue
        questions = problem.find_all(class_='oli-question')
        for question in questions:
            try:
                qid = question.find(class_="oli-question").get_text()
            except:
                continue # there are empty oli-question class objects. Skipping them.
            question_data = {
                'file_path':str(file_path),
                'problem': problem_name,
                'qid': qid,
                'steps': [], # steps
                'parts': [], # feedback corresponds to each step.
                'images': [],
            } # objects must have key_str for mapping location: {'key_str': '[|str|]'} 
            
            # process parts first, since need to recover missing steps from parts.
            parts = question.find_all(class_='oli-part')
            for part in parts:
                part_id = part.get('id')
                part_input = part.get('input')
                part_obj = {'part_id':part_id,'responses':[],'input':part_input}
                if part.find('title'):
                    part_obj['title'] = part.find('title').get_text()

                responses = part.find_all(class_='oli-response') + part.find_all(class_='oli-no-response')
                for response in responses:                
                    part_obj['input'] = response.get('input', '') if not part_obj['input'] else part_obj['input']
                    feedback = {
                        'class': response.get('class',''),
                        'input': response.get('input', ''),
                        'match': response.get('match', ''),
                        'name': response.get('name', ''),
                        'score': response.get('score', ''),
                    }

                    # Handling both visible and commented-out feedback
                    if response.find('feedback'):
                        feedback['feedback'] = response.find('feedback').get_text(strip=True)
                    else:
                        comment = response.find(string=lambda text: isinstance(text, Comment))
                        if comment:
                            feedback_comment = BeautifulSoup(comment, 'html.parser')
                            feedback['text'] = feedback_comment.get_text(strip=True)

                    part_obj['responses'].append(feedback)
                if part.find(class_='oli-hint'):
                    part_obj['hints'] = [h.get_text() for h in part.find_all(class_='oli-hint')]

                question_data['parts'].append(part_obj)

            oli_body = question.find(class_='oli-body')
            if oli_body:

                for img in oli_body.find_all('img'):
                    image_key_str = f"[|image_{len(question_data['images'])}|]"
                    question_data['images'].append({'key_str':image_key_str,'text':img['src']})
                    img.replace_with(f"\n\n{image_key_str}\n\n")

                for obj in question.find_all('object', type='application/x-shockwave-flash'):
                    replace_str = ""
                    if obj.has_attr('data'):
                        swf_path = os.path.normpath(os.path.join(data_root, 'resources' , os.path.basename(obj['data'])))
                        swf_name = os.path.basename(swf_path).split('.swf')[0]
                        obj_keys = []
                        for r in os.listdir(os.path.dirname(swf_path)):                        
                            if r.startswith(swf_name) and not r.endswith('.swf'):
                                parsed_image_path = os.path.join(os.path.dirname(obj['data']),r)                            
                                image_key_str = f"[|image_{len(question_data['images'])}|]"
                                question_data['images'].append({'key_str':image_key_str,'text':parsed_image_path})
                                obj_keys.append(image_key_str)
                        if len(obj_keys) > 0:
                            replace_str = f"\n\n[|images extracted from flash|]: {' '.join(obj_keys)}\n\n"
                        
                        audio_param = obj.find('param', {'name': 'audio_file'})
                        if audio_param:
                            mp3_path_tag = audio_param.find_next('path')
                            if mp3_path_tag and mp3_path_tag.has_attr('href'):
                                mp3_path = os.path.join(os.path.dirname(swf_path),os.path.basename(mp3_path_tag['href']))
                                    
                                txt_path = mp3_path.replace('.mp3','.txt')                                
                                if os.path.exists(txt_path.strip()):
                                    with open(txt_path.strip(), 'r', encoding='utf-8') as txt_file:
                                        transcription = txt_file.read().strip()
                                        replace_str += f"\n[transcription of embedded mp3 file]: {transcription}\n\n"
                                    # print(transcription)

                        obj.replace_with(replace_str)
                        
                all_steps = []
                all_steps.extend(question.find_all(class_='oli-multiple-choice'))
                all_steps.extend(question.find_all(class_='oli-text'))
                all_steps.extend(question.find_all(class_='oli-fill-in-the-blank'))
                all_steps.extend(question.find_all(class_='oli-numeric'))
                all_steps.extend(question.find_all(class_='oli-short-answer'))
                all_steps.extend(question.find_all(class_='oli-image-hotspot'))

                all_steps.sort(key=lambda x: x.sourceline)
                for step in all_steps:
                    step_class = step['class'][0]
                    step_id = step['input'] if step.get('input') else step.get('id')
                    if step_class == 'oli-multiple-choice':
                        step_obj = {
                            'step_id': step_id,
                            'key_str': f"[|multiple_choice_{step_id}|]",
                            'step_type': 'oli-multiple-choice',
                            'options': [],
                        }
                        for op in step.find_all('option') + step.find_all(class_='oli-choice'):
                            op_obj = {'text': op.get_text()}
                            if op.get('value'):
                                op_obj['value'] = op['value']
                            if op.get('id'):
                                op_obj['id'] = op['id']
                            step_obj['options'].append(op_obj)

                        question_data['steps'].append(step_obj)                
                        step.replace_with(f"[|multiple_choice_{step_id}|]")


                    elif step_class == 'oli-text':
                        question_data['steps'].append({
                            'step_id': step_id,
                            'step_type': 'oli-text',
                            'key_str': f"[|type_answer_{step_id}|]",
                        })
                        step.replace_with(f"[|type_answer_{step_id}|]")

                    elif step_class == 'oli-fill-in-the-blank':
                        question_data['steps'].append({
                            'step_id': step_id,
                            'step_type': 'oli-fill-in-the-blank',
                            'options': [{'value':o['value'], 'text':o.get_text()} for o in step.find_all('option')],
                            'key_str': f"[|fill_blank_{step_id}|]",
                        })
                        step.replace_with(f"[|fill_blank_{step_id}|]")

                    elif step_class == 'oli-numeric':
                        question_data['steps'].append({
                            'step_id': step_id,
                            'step_type': 'oli-numeric',
                            'key_str': f"[|fill_numeric_{step_id}|]",
                        })
                        step.replace_with(f"[|fill_numeric_{step_id}|]")

                    elif step_class == 'oli-short-answer':
                        question_data['steps'].append({
                            'step_id': step_id,
                            'step_type': 'oli-short-answer',
                            'key_str': f"[|fill_blank_{step_id}|]",
                        })
                        step.replace_with(f"[|fill_blank_{step_id}|]")

                    elif step_class == 'oli-image-hotspot':
                        step_obj = {
                            'step_id': step_id,
                            'step_type': 'oli-image-hotspot',
                            'key_str': f"[|hotspot_image_{step_id}|]",
                        }
                        if step.find('area'):
                            step_obj['options'] = [o.get('title') for o in step.find_all('area')]
                        if step.find(class_="oli-image-hotspot-note"):
                            step_obj['note'] = step.find(class_="oli-image-hotspot-note").get_text()
                        question_data['steps'].append(step_obj)
                        step.replace_with(f"[|hotspot_image_{step_id}|]")

                # Backup hotspot to cover the data missing in _m1_assess, _m12_assess
                if not question_data['steps'] and question_data['parts']:
                    for part in question_data['parts']:                    
                        step_id = part['part_id']
                        step_obj = {
                            'step_id': step_id,
                            'step_type': 'oli-image-hotspot',
                            'key_str': f"[|hotspot_image_{step_id}|]",
                        }
                        if part['responses']:
                            step_obj['options'] = [ r.get('match') for r in part['responses']]

                        question_data['steps'].append(step_obj)                    
                        oli_body.append(f"[|hotspot_image_{step_id}|]")
                
                question_text = oli_body.get_text()
                
                # step-part mapping part
                question_data['question'] = clean_text(question_text)
                for step_idx, step_obj in enumerate(question_data['steps']):
                    if not step_obj['step_id']:
                        step_obj['step_id'] = f"step_{step_idx+1}"
                if len(question_data['steps']) == len(question_data['parts']):
                    # if exact mapper exist, map by id or input field. Otherwise, just match by order.
                    if set([i['step_id'] for i in question_data['steps']]) == set([i['input'] for i in question_data['parts']]):
                        for part_obj in question_data['parts']:
                            part_obj['step_id'] = part_obj['input']
                    elif set([i['step_id'] for i in question_data['steps']]) == set([i['part_id'] for i in question_data['parts']]):
                        for part_obj in question_data['parts']:
                            part_obj['step_id'] = part_obj['part_id']
                    else:
                        for step_obj, part_obj in zip(question_data['steps'], question_data['parts']):
                            part_obj['step_id'] = step_obj['step_id']
                else:
                    print(f"Question {qid} of Problem {problem_name} is not matched with feedback.")
                    continue
                    # raise ValueError('shit')
                    

            else:
                raise ValueError('Body is not found')
        
            
            problem_parsed.append(question_data)
            
    return problem_parsed

def parse_html(data_root, file_path, valid_problems):
    with open(file_path, 'r', encoding='utf-8') as file:
        return extract_questions(data_root, file_path, BeautifulSoup(file.read(), 'html.parser'),valid_problems)


def get_datashop_transaction(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if 'tx_All_Data' in file and file.endswith('.txt'):
                return pd.read_csv(os.path.join(root, file), delimiter='\t', low_memory=False)
            
    raise ValueError(f'All Transaction Data is not observed on the desired path: {directory_path}')

def traverse_and_parse(data_path, valid_problems):
    parsed_data = []
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.html'):
                file_path = os.path.join(root, file)
                parsed = parse_html(data_path, file_path, valid_problems)
                parsed_data += parsed
    
    return parsed_data
    
if __name__ == '__main__':
    content_path = Path('./')
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
    args = parser.parse_args()
    target_dsets = [args.dataset] if args.dataset != 'all' else dataset_choices
    for dset in target_dsets:
        print(f'Processing {dset}..')
        datashop_data = get_datashop_transaction(os.path.join(content_path,dset_config[dset]['transaction_path']))
        valid_problems = set([x['Problem Name'] for _,x in datashop_data.iterrows()])
        
        data_path_list = [ os.path.join(content_path, dset_config[dset]['content_data_path'], subject_name) for subject_name in dset_config[dset]['subjects']]
        
        parsed_data = []
        for data_path in data_path_list:
            parsed_data = parsed_data + traverse_and_parse(data_path, valid_problems)
        
        # print(f'From entire {len(datashop_data)} transactions, # {len()} were matched with the content data.')
        
        problems = set()
        for i in parsed_data:
            problems.add(i['problem'])
        print(f'Unmatched_problems: {len(valid_problems - problems)} / Original problems: {len(valid_problems)}')
        
        json.dump(parsed_data,open(os.path.join(content_path, 'resources' ,dset,'parsed_steps.json'),'w'))
        
        
        batch_list = parsed_data2batch_list(parsed_data)
        with open(os.path.join(content_path, 'resources' ,dset,'batch_input.jsonl'), 'w') as file:
            for b in batch_list:
                json_b = json.dumps(b)
                file.write(json_b + '\n')
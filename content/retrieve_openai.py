import json

def parse_jsonl(file):
    data_list = [json.loads(line) for line in file.split('\n') if line.strip()]
    return data_list

if __name__ == '__main__':
    content_path = Path('./')
    jobs = json.load(open(content_path / 'batch_info.json'))
    job_ids = [ job['id'] for job in jobs ]

    for dset, job_id in zip(dset_config.keys(), job_ids):
        batch = client.batches.retrieve(job_id)
        print(dset, batch.status)
        success = client.files.content(batch.output_file_id)
        success_list = parse_jsonl(success.text)
        with open(os.path.join(content_path, 'resources', dset, 'retry_gpt4_success.jsonl'), 'w') as file:
            for l in success_list:
                json_l = json.dumps(l)
                file.write(json_l + '\n')

        if batch.error_file_id:
            failure = client.files.content(batch.error_file_id)
            failure_list = parse_jsonl(failure.text)
            with open(os.path.join(content_path, 'resources', dset, 'retry_gpt4_failure.jsonl'), 'w') as file:
                for l in failure_list:
                    json_l = json.dumps(l)
                    file.write(json_l + '\n')
        else:
            failure_list = []
        print(f'For {dset}, total success: {len(success_list)}, failure: {len(failure_list)}')

import os
import json
from pathlib import Path
from openai import OpenAI
from utils import *

if __name__ == '__main__':

    client = OpenAI(api_key = OPENAI_KEY) # stored in utils.
    
    content_path = Path('./')
    dset_config = json.load(open(os.path.join(content_path,'config.json')))

    jobs = []
    for dset in dset_config.keys():
        batch_input_file = client.files.create(
            file=open(f"resources/{dset}/batch_input.jsonl", "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        created_job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
              "description": f"{dset} full inference"
            }
        )
        jobs.append(created_job.to_dict())

    json.dump(jobs, open(content_path / 'batch_info.json','w'))
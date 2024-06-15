# Preparing the dataset

To reproduce the experiment, there are two options.
1. Download preprocessed data from CMU datashop (recommanded)
2. Preprocess data on your own

We recommand the first option since step 2 requires extracting flash files.

You need to request access to these datasets.

[OLI Engineering Statics](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507)
[OLI Principles of Computing](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1806)
[OLI French](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=918)
[OLI Biology](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1148)
[OLI Psychology](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=863)

## 1. Download preprocessed data



## 2. Reproduce preprocessing procedure

1. Download content data and transaction data from the CMU datashop. After you get dataset access, download datas and modify the 'config.json' to their path. Placing the unzipped files into /resources/{Name of Dataset} is recommanded and is the defaut option.


2. extract image files from swf files. Make sure that all content data are located under resources folder.

```bash
./resources/all_parse.cmd
```

To run this, an swf extractor, which can parse flash files into image files, is required. In our case, we used swfextract.exe obtained from http://www.swftools.org/swftools-0.9.2.tar.gz, which requires Window OS. 

3. run preprocessing 

```bash
python parse_data.py all
```
This will produce 'resources/{dataset_name}/parsed_data.json' and 'resources/{dataset_name}/batch_input.jsonl'. The first file is a parsed content data, which allows you to use them to your further exploration. The jsonl file is a format that supports OpenAI batch inference. 

4. Inference GPT by openai API.

```bash
python openai_batch.py
```
This will generate files for openai batch inference.

```bash
python infer_openai.py
```

You can modify infer_openai.py and use your preferred multimodal LLMs. In our case, we used GPT-4o with batch inference. Which costs about 40$ in total.
Note that some of the resources violate OPENAI content policy, with unknown reason. For these failure cases, we will load all images from the respective step using the PIL library and save them again for retry. In such cases, the alpha channel of the images gets lost, resulting in some degradation. However, it has been observed that GPT can still recognize the images. 

```bash
python reterive_openai.py
```

This will recieve infernece result.

5. Obtain text encoding
For clustering, run
```python
python kc_clustering.py all openai_3
```

# Reproduce experiments

```bash
python create_sil_test_dset.py
```
this will generate test file for AFM clustering test.

```bash
python generate_datashop_form.py
```
this will generate cmu-datashop form transaction files, and also roll-up file.

```bash
python generate_kt_tsv.py Ours
```
this will convert cmu datashop form into the format of kt_benchmark repo. 
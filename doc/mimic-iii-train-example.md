## Using MIMIC-III Data Training ICD10 Classification Model
### Preparation - Get Raw MIMIC-III Data
Suppose you put original MIMIC-III data under `./_data/raw/mimic3/` like:
```
./_data/raw/mimic3/
├── DIAGNOSES_ICD.csv
├── NOTEEVENTS.csv
└── PROCEDURES_ICD.csv

0 directories, 3 files
```
### ETL - Training Dataset Building
This is about join necessary tables' data together and build training dataset. Suppose we are 
going to put training data under `./_data/etl/mimic3/`, as this programed rules, the directory 
should contain 3 files, train.jsonl, dev.jsonl and test.jsonl, like:
```
./_data/etl/mimic3/
├── dev.jsonl
├── dict.json
├── dim_processed_base_data.jsonl
├── test.jsonl
└── train.jsonl

0 directories, 5 files
```
You can run:
```shell
python ./bin/etl/etl_mimic3_processing.py ./_data/raw/mimic3/ ./_data/etl/mimic3/ 
```

### Config - Prepare Your Training Config File
The `data_dir` in this config will be needed by next ETL step, can just refer to `train_mimic3_icd.json`.

### ETL - Convert Training Dataset JSONL to Limited JSONL File
Note this step is unnecessary, since the outputs of `./bin/etl/etl_mimic3_processing.py` have 
already been limited JSON line files, so even though you run following program, you will get 
exactly same files:
```shell
python ./bin/etl/etl_jsonl2limited_jsonl.py ./_data/raw/mimic3/${INPUT_JSONL_FILE} ./_data/raw/mimic3/${OUTPUT_JSONL_FILE}
```

### Training - Training ICD10 Classification Model with MIMIC-II Dataset
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./train.py ./train_mimic3_icd.json
```



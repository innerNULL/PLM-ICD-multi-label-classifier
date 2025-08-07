## MIMIC3 Dataset ETL
The ETL contain following steps:
* Origin JSON line dataset preparation
* Transform JSON line file to **limited** JOSN line file, which means all `list` or `dict` 
  will be transformed to `string`.
* Data dictionary generation.

Note, the final data folder should contains 4 files: train.jsonl, dev.jsonl, test.jsonl, dict.json.

### Prepare (Specific) Original JSON Line Dataset
The data should be in JSON line format, here provide an MIMIC-III data ETL program:
```sh
python ./bin/etl/etl_mimic3_processing.py ${YOUR_MIMIC3_DATA_DIRECTORY} ${YOUR_TARGET_OUTPUT_DIRECTORY}
```
When you need use this program do text multi-label classification on your custimized 
data set, you can just transfer it into a JSON line file, and using **training config** 
file to specify which field is text and which is label. 

**NOTE**, since here you are dealing a multi-label classification task, the format of 
label field should be as a CSV string, for example:
```
{"text": "this is a fake text.", "label": "label1,label2,label3,label4"}
```

But you can also use your specific dataset.

### Transform To Limited JSON Line Dataset
Although using JSON line file, here do not allow `list` and `dict` contained in JOSN. 
I believe "flat" JSON can make things clear, so here provide a tool which can help 
to convert `list` and `dict` contained in JSON to `string`:
```shell
python ./bin/etl/etl_jsonl2limited_jsonl.py ${ORIGINAL_JSON_LINE_DATASET} ${TRANSFORMED_JSON_LINE_DATASET}
```

**NOTE, alghouth you can put dataset in anly directory you like, but you HAVE TO naming you datasets 
as train.jsonl, dev.jsonl and test.jsonl.**

### Data Dictionary Generation
Generate (some) data dictionaries by scanning train, dev and test data. Run:
```shell
python ./bin/etl/etl_generate_data_dict.py ${TRAIN_CONFIG_JSON_FILE_PATH}
```



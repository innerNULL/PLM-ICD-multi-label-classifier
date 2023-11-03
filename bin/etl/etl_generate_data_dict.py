# -*- coding: utf-8 -*-
# file: etl_generate_data_dict.py
# date: 2023-10-31


import pdb
import os
import sys
import json
import duckdb
from typing import Dict, List
from duckdb import DuckDBPyRelation
from pandas import DataFrame


def generate_label_dict(
    all_data_path: str, out_dir: str, label_col: str="label"
) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    label2id: Dict[str, int] = {}
    id2label: Dict[int, str] = {}

    label_set: Set[str] = set()

    labels: List[str] = duckdb.query(
        "select distinct %s from read_json_auto(%s);" 
        % (label_col, all_data_path)
    ).df().iloc[:, 0].tolist()
    for label in labels:
        codes: List[str] = label.strip("\n").split(",")
        for code in codes:
            label_set.add(code)

    for i, label in enumerate(sorted(list(label_set))):
        label2id[label] = i
        id2label[i] = label
    
    out["label2id"] = label2id
    out["id2label"] = id2label
    out_file_path: str = os.path.join(out_dir, "dict.json")
    open(out_file_path, "w").write(json.dumps(out))


if __name__ == "__main__":
    train_conf: Dict = json.loads(open(sys.argv[1], "r").read())
    label_col: str = train_conf["label_col"]
    data_root_path: str = train_conf["data_dir"]
    dataset_paths: List = [
        os.path.join(data_root_path, "train.jsonl"), 
        os.path.join(data_root_path, "dev.jsonl"),
        os.path.join(data_root_path, "test.jsonl")
    ]
    dataset_paths = [x for x in dataset_paths if os.path.exists(x)]
    all_data_paths: str = json.dumps(dataset_paths).replace("\"", "'")
    print("Using following data to build data dictionary: \n%s" % all_data_paths)

    generate_label_dict(all_data_paths, data_root_path, label_col)



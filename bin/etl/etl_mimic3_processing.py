# -*- coding: utf-8 -*-
# file: etl_mimic3_processing.py
# date: 2023-09-22
#
# An example usage:
# python etl_mimic3_processing.py ./_data/raw_data/mimic3/ ./_data/etl/mimic3


import pdb
import os
import sys
import json
import duckdb
from typing import Dict, Tuple
from duckdb import DuckDBPyRelation
from pandas import DataFrame


PROCESSED_DATA_FILE: str = "dim_processed_base_data.jsonl"


def init(out_data_dir: str) -> None:
    os.system("mkdir -p %s" % out_data_dir)


def processing(src_data_dir: str, out_data_path: str) -> None:
    # MIMIC-III raw data paths
    procedures_data_path: str = os.path.join(src_data_dir, "PROCEDURES_ICD.csv")
    diagnosis_data_path: str = os.path.join(src_data_dir, "DIAGNOSES_ICD.csv")
    noteevents_data_path: str = os.path.join(src_data_dir, "NOTEEVENTS.csv")

    # Raw data
    dim_procedures: DuckDBPyRelation = duckdb.query(
        """
        select * from read_csv('%s', AUTO_DETECT=TRUE) 
        where ICD9_CODE is not NULL and len(ICD9_CODE) > 0; 
        """ % procedures_data_path
    )
    dim_diagnosis: DuckDBPyRelation = duckdb.query(
        """
        select * from read_csv('%s', AUTO_DETECT=TRUE) 
        where ICD9_CODE is not NULL and len(ICD9_CODE) > 0; 
        """ % diagnosis_data_path
    )
    dim_noteevents: DuckDBPyRelation = duckdb.query(
        "select * from read_csv('%s', AUTO_DETECT=TRUE);" 
        % noteevents_data_path
    )

    # Transformed data
    dim_log_icd: DuckDBPyRelation = duckdb.query(
        """
        select * from dim_procedures union select * from dim_diagnosis;
        """
    )
    agg_icd_list: DuckDBPyRelation = duckdb.query(
        """
        select 
        SUBJECT_ID, HADM_ID, string_agg(ICD9_CODE, ',') as icds, 
        count(1) as cnt 
        from dim_log_icd group by SUBJECT_ID, HADM_ID;
        """
    )
    dim_full_data: DuckDBPyRelation = duckdb.query(
        """
        select t1.*, t2.TEXT 
        from agg_icd_list as t1 inner join dim_noteevents as t2 
        on t1.SUBJECT_ID = t2.SUBJECT_ID and t1.HADM_ID = t2.HADM_ID;
        """
    )
    dim_full_data.df().to_json(
        os.path.join(out_data_dir, PROCESSED_DATA_FILE), orient="records"
    )


def train_data_gen(
    src_data_path: str, out_data_dir: str, 
    train_data_ratio: float=0.98, dev_data_ratio: float=0.01
) -> None:
    assert(train_data_ratio + dev_data_ratio <= 0.99)

    full_data: DuckDBPyRelation = duckdb.query(
        """
        select setseed(0.16);
        select TEXT as text, icds from read_json('%s', AUTO_DETECT=TRUE)
        order by random() asc; 
        """ % src_data_path
    )
    total_data_size: int = duckdb.query("select count(1) from full_data;").df().iloc[0, 0]
    train_data_size: int = int(total_data_size * train_data_ratio)
    dev_data_size: int = int(total_data_size * dev_data_ratio)
    test_data_size: int = total_data_size - train_data_size - dev_data_size

    train_data_start_idx: int = 0
    train_data_end_idx: int = train_data_size
    dev_data_start_idx: int = train_data_size
    dev_data_end_idx: int = dev_data_start_idx + dev_data_size
    test_data_start_idx: int = dev_data_end_idx
    test_data_end_idx: int = test_data_start_idx + test_data_size
    
    train_data: DataFrame = full_data.df().iloc[train_data_start_idx:train_data_end_idx, :]
    dev_data: DataFrame = full_data.df().iloc[dev_data_start_idx:dev_data_end_idx, :]
    test_data: DataFrame = full_data.df().iloc[test_data_start_idx:test_data_end_idx, :]
    
    train_data.to_json(os.path.join(out_data_dir, "train.jsonl"), orient="records")
    dev_data.to_json(os.path.join(out_data_dir, "dev.jsonl"), orient="records")
    test_data.to_json(os.path.join(out_data_dir, "test.jsonl"), orient="records")


def generate_label_dict(
    full_data_path: str, out_dir: str, label_col: str="label"
) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    label2id: Dict[str, int] = {}
    id2label: Dict[int, str] = {}

    label_set: Set[str] = set()

    labels: List[str] = duckdb.query(
        "select distinct %s from read_json('%s', AUTO_DETECT=TRUE);" 
        % (label_col, full_data_path)
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
    src_data_dir: str = sys.argv[1]
    out_data_dir: str = sys.argv[2]

    full_base_data_path: str = os.path.join(out_data_dir, PROCESSED_DATA_FILE)

    init(out_data_dir)

    if not os.path.exists(full_base_data_path):
        processing(src_data_dir, full_base_data_path)
    else:
        print("Using local cache: %s" % full_base_data_path)

    if not os.path.exists(os.path.join(out_data_dir, "train.jsonl")):
        train_data_gen(full_base_data_path, out_data_dir)
    else:
        print("Using local train/dev/test data cache")
    
    if not os.path.exists(os.path.join(out_data_dir, "dict.json")):
        generate_label_dict(full_base_data_path, out_data_dir, "icds")
    else:
        print("Using local cache: %s" % os.path.join(out_data_dir, "dict.json"))

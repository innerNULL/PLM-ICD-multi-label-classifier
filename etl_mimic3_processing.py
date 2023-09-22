# -*- coding: utf-8 -*-
# file: etl_mimic3_processing.py
# date: 2023-09-22
#
# An example usage:
# python etl_mimic3_processing.py ./_data/raw_data/mimic-iii-clinical-database-demo-1.4/ ./_data/trans/mimic3


import pdb
import os
import sys
import duckdb
from duckdb import DuckDBPyRelation


def init(out_data_dir: str) -> None:
    os.system("mkdir -p %s" % out_data_dir)


def etl(src_data_dir: str, out_data_dir: str) -> None:
    # MIMIC-III raw data paths
    procedures_data_path: str = os.path.join(src_data_dir, "PROCEDURES_ICD.csv")
    diagnosis_data_path: str = os.path.join(src_data_dir, "DIAGNOSES_ICD.csv")
    noteevents_data_path: str = os.path.join(src_data_dir, "NOTEEVENTS.csv")

    # Raw data
    dim_procedures: DuckDBPyRelation = duckdb.query(
        "select * from read_csv('%s', AUTO_DETECT=TRUE);" 
        % procedures_data_path
    )
    dim_diagnosis: DuckDBPyRelation = duckdb.query(
        "select * from read_csv('%s', AUTO_DETECT=TRUE);" 
        % diagnosis_data_path
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
    dim_full_data.df().to_csv(
        os.path.join(out_data_dir, "dim_full_data.csv"), header=True
    )


if __name__ == "__main__":
    src_data_dir: str = sys.argv[1]
    out_data_dir: str = sys.argv[2]

    init(out_data_dir)
    etl(src_data_dir, out_data_dir)

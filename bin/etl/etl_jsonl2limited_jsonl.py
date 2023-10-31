# -*- coding: utf-8 -*-
# file: etl_jsonl2limited_jsonl.py
# date: 2023-10-31


import pdb
import sys
import traceback
import json
from typing import Dict, Union


def json2limited_json(norm_json: Union[str, Dict]) -> str:
    out: Dict = {}
    
    if isinstance(norm_json, str):
        norm_json = json.loads(norm_json)

    for key in norm_json:
        if isinstance(norm_json[key], list):
            out[key] = ",".join([str(x) for x in norm_json[key]])
        elif isinstance(norm_json[key], dict):
            out[key] = json.dump(norm_json[key])
        else:
            out[key] = norm_json[key]

    return json.dumps(out)


def jsonl2limited_jsonl(input_jsonl_path: str, output_jsonl_path: str) -> None:
    input_jsonl_file = open(input_jsonl_path, "r")
    output_jsonl_file = open(output_jsonl_path, "w")

    cnt: int = 0
    err_cnt: int = 0
    json_line: str = input_jsonl_file.readline()
    while json_line:
        cnt += 1
        try:
            limited_json: Dict = json2limited_json(json_line)
            output_jsonl_file.write(limited_json + "\n")
            json_line = input_jsonl_file.readline()
        except Exception as e:
            err_cnt += 1
            print(traceback.format_exc())
            print(json_line)
            

    input_jsonl_file.close()
    output_jsonl_file.close()

    print("Transfered and dump new data to '%s'" % output_jsonl_path)
    print("%i / %i records are failed to transform" % (err_cnt, cnt))


if __name__ == "__main__":
    input_jsonl_path: str = sys.argv[1]
    output_jsonl_path: str = sys.argv[2]

    jsonl2limited_jsonl(input_jsonl_path, output_jsonl_path)


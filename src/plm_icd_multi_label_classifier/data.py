# -*- coding: utf-8 -*-
# file: data.py
# date: 2023-09-23


import pdb
import traceback
import json
import duckdb
import torch
import pandas as pd
from typing import List, Dict, Tuple
from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from . import text
from .model_ctx import PlmIcdCtx


class TextOnlyDataset(Dataset):
    def __init__(self, 
        data_path: str, data_dict_path: str, tokenizer: AutoTokenizer,
        text_col: str="text", label_col: str="label", data_format: int="csv", 
        chunk_size: int=512, chunk_num: int=2
    ):
        self.data_path: str = data_path
        self.data_dict: Dict[str, Dict] = json.loads(open(data_dict_path, "r").read())
        self.text_col: str = text_col
        self.label_col: str = label_col
        self.data: List[Dict] = []
        self.model_ctx: PlmIcdCtx = PlmIcdCtx().init(
            data_dict_path=data_dict_path, lm_tokenizer=tokenizer, 
            chunk_size=chunk_size, chunk_num=chunk_num
        )

        if data_format == "csv":
            self.data = pd.read_csv(data_path)[[text_col, label_col]].to_dict(orient="records")
        elif data_format in {"jsonl", "json"}:
            self.data = duckdb.query(
                "select %s, %s from read_json_auto('%s');" % (text_col, label_col, data_path)
            ).df()[[text_col, label_col]].to_dict(orient="records")
        
        # Remove unseem label in label dictionary, if not may raise error when
        # using cutomized dev/test data do evaluation.
        for i, record in enumerate(self.data):
            curr_filtered_label: List[str] = [
                x for x in record[label_col].split(",") if x in self.data_dict["label2id"]
            ]
            if len(curr_filtered_label) == 0:
                self.data[i] = None
        self.data = [x for x in self.data if x is not None]
        assert(len(self.data) > 0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[LongTensor, LongTensor, FloatTensor]:
        model_inputs: Dict[str, Tensor] = self.model_ctx.json_inputs2model_train_inputs(
            json_inputs=self.data[idx], 
            text_fields=[self.text_col], label_field=self.label_col
        )
        return (
            model_inputs["token_ids"], model_inputs["attn_masks"], 
            model_inputs["label_one_hot"]
        )

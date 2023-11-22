# -*- coding: utf-8 -*-
# file: model_ctx.py
# date: 2023-11-05


import os
import json
import torch
from typing import Union, Dict, Optional, List
from transformers import AutoTokenizer
from torch import Tensor, FloatTensor, LongTensor

from . import text


class PlmIcdCtx():
    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.data_dict: Dict[str, Dict] = {}
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}
        self.chunk_size: int = -1
        self.chunk_num: int = -1

    def init_by_train_config(self, train_conf_path: str):
        train_conf: Dict = json.loads(open(train_conf_path, "r").read())
        data_dict_path: str = os.path.join(train_conf["data_dir"], "dict.json")
        lm_tokenizer: str = train_conf["hf_lm"]
        chunk_size: int = train_conf["chunk_size"] 
        chunk_num: int = train_conf["chunk_num"]

        return self.init(data_dict_path, lm_tokenizer, chunk_size, chunk_num)

    def init(self, 
        data_dict_path: str, 
        lm_tokenizer: Union[str, AutoTokenizer], 
        chunk_size: int, chunk_num: int
    ):
        self.data_dict = json.loads(open(data_dict_path, "r").read())
        self.id2label = {int(k): v for k, v in self.data_dict["id2label"].items()}
        self.label2id = self.data_dict["label2id"]
        self.tokenizer = \
            AutoTokenizer.from_pretrained(lm_tokenizer) if isinstance(lm_tokenizer, str) \
            else lm_tokenizer
        self.chunk_size = chunk_size
        self.chunk_num = chunk_num
        return self

    def json_inputs2model_inf_inputs(self, 
        json_inputs: Union[str, Dict], text_fields: List[str]
    ) -> Dict[str, Tensor]:
        text_input: str = " ".join([json_inputs[x] for x in text_fields])
        token_ids: List[List[int]] = []
        token_masks: List[List[int]] = []

        token_ids, attn_masks = text.tokenize_to_chunks(
            text_input, self.tokenizer, self.chunk_size, self.chunk_num
        )
        return {"token_ids": LongTensor([token_ids]), "attn_masks": LongTensor([attn_masks])}

    def json_inputs2model_train_inputs(self, 
        json_inputs: Union[str, Dict], text_fields: List[str], label_field: str
    ) -> Dict[str, Tensor]:
        model_inputs: Dict[Tensor] = self.json_inputs2model_inf_inputs(
            json_inputs, text_fields
        )
        label_names: List[str] = json_inputs[label_field].split(",")
        label_ids: List[int] = [
            self.label2id[x] for x in label_names if x in self.label2id
        ]
        label_one_hot: FloatTensor = torch.zeros(len(self.label2id))
        for label_id in label_ids:
            label_one_hot[label_id] = 1.0

        assert(label_one_hot.sum() >= 1)
        model_inputs["label_one_hot"] = label_one_hot
        model_inputs["token_ids"] = torch.squeeze(model_inputs["token_ids"], 0)
        model_inputs["attn_masks"] = torch.squeeze(model_inputs["attn_masks"], 0)
        
        return model_inputs

    def model_outputs2json_outputs(self, 
        model_outputs: Dict[str, Tensor], logits_field: str="logits"
    ) -> Dict[str, float]:
        """
        Not support batch inference outputs right now.
        """
        logits: FloatTensor = model_outputs[logits_field]

        assert(len(logits.shape) <= 2)
        logits = logits[0] if len(logits.shape) == 2 else logits

        scores: FloatTensor = torch.sigmoid(logits)

        outputs: Dict[str, float] = {}
        for i in range(scores.shape[0]):
            label: str = self.id2label[i] 
            outputs[label] = float(scores[i])
        return outputs

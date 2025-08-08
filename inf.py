# -*- coding: utf-8 -*-
# file: eval.py
# date: 2025-08-04


import pdb
import os
import sys
import json
import torch
from typing import Dict, List
from transformers import AutoTokenizer
from torch import Tensor
from tqdm import tqdm

from src.plm_icd_multi_label_classifier import text
from src.plm_icd_multi_label_classifier.model import PlmMultiLabelEncoder
from src.plm_icd_multi_label_classifier.data import TextOnlyDataset
from src.plm_icd_multi_label_classifier.model_ctx import PlmIcdCtx
from src.plm_icd_multi_label_classifier.metrics import metrics_func, topk_metrics_func
from src.plm_icd_multi_label_classifier.eval import evaluation


def main() -> None:
    configs: Dict = json.load(open(sys.argv[1], "r"))
    print(json.dumps(configs, indent=2))

    model_conf: Dict = configs["model"]
    inf_conf: Dict = configs["inf"]
    
    samples: List[Dict] = [
        json.loads(x) for x in 
        open(configs["test_data_path"], "r").read().split("\n")
        if x not in {""}
    ]
    label_dict: Dict = json.load(open(configs["label_dict_path"], "r"))
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_conf["hf_lm"])
    model: PlmMultiLabelEncoder = PlmMultiLabelEncoder(
        len(label_dict["label2id"]), 
        model_conf["hf_lm"],
        model_conf["lm_hidden_dim"],
        model_conf["chunk_size"], 
        model_conf["chunk_num"]
    )
    model_ctx: PlmIcdCtx = PlmIcdCtx()

    model.load_state_dict(torch.load(configs["ckpt_path"]))
    model_ctx.init(
        configs["label_dict_path"],
        tokenizer,
        model_conf["chunk_size"],    
        model_conf["chunk_num"],
        inf_conf["label_splitter"]
    )

    out_file = open(configs["out_path"], "w")
    for sample in tqdm(samples):
        model_inputs: Dict[str, Tensor] = model_ctx.json_inputs2model_inf_inputs(
            json_inputs=sample,
            text_fields=[configs["text_col"]]
        )
        model_output: Tensor = model.forward(
            token_ids=model_inputs["token_ids"],
            attn_masks=model_inputs["attn_masks"]
        )
        pred_outputs: Dict[str, float] = model_ctx.model_outputs2json_outputs(
            {"logits": model_output}, 
            "logits"
        )
        label_scores: List[Tuple[str, float]] = sorted(
            [(k, v) for k, v in pred_outputs.items()],
            reverse=True, 
            key=lambda x: x[1]
        )
        label_scores = [
            x for x in label_scores[:inf_conf["top_k"]] 
            if x[1] >= inf_conf["min_confidence"]
        ]
        pred_outputs = {x[0]: x[1] for x in label_scores}
        sample[configs["result_col"]] = pred_outputs
        out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    out_file.close()
    print("Dumped to %s" % configs["out_path"])
    return


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# file: eval.py
# date: 2025-08-05


import pdb
import os
import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch import IntTensor

from src.plm_icd_multi_label_classifier.metrics import metrics_func


def verification(
    train_eval_results_path: Optional[str],
    eval_gt_one_hots: List[List[int]],
    eval_pred_one_hots: List[List[int]]
) -> None:
    if train_eval_results_path is None:
        return
    train_eval_results: Dict = json.load(open(train_eval_results_path, "r")) 
    train_gt_one_hots: List[List[int]] = train_eval_results["verbose"]["gt_one_hot"]
    train_pred_one_hots: List[List[int]] = train_eval_results["verbose"]["pred_one_hot"] 
    assert(len(train_gt_one_hots) == len(train_pred_one_hots))
    assert(len(train_pred_one_hots) == len(eval_gt_one_hots))
    assert(len(eval_gt_one_hots) == len(eval_pred_one_hots))
    for i in range(len(train_gt_one_hots)):
        train_gt_one_hot: List[int] = train_gt_one_hots[i]
        eval_gt_one_hot: List[int] = eval_gt_one_hots[i]
        assert(len(train_gt_one_hot) == len(eval_gt_one_hot))
        for j in range(len(eval_gt_one_hot)):
            assert(train_gt_one_hot[j] == eval_gt_one_hot[j])
    return


def main() -> None:
    configs: Dict = json.load(open(sys.argv[1], "r"))
    print(json.dumps(configs, indent=2))
    label_dict_path: str = configs["label_dict_path"]
    gt_label_col: str = configs["gt_label_col"]
    pred_results_col: str = configs["pred_results_col"]
    min_confidence: float = configs["min_confidence"]
    label_splitter: str = configs["label_splitter"]
    train_eval_results_path: Optional[str] = configs["train_eval_results_path"]

    label_dict: Dict = json.load(open(label_dict_path, "r"))
    inf_results: List[Dict] = [
        json.loads(x) 
        for x in open(configs["inf_results_path"], "r").read().split("\n")
        if x not in {""}
    ]

    label_dim: int = len(label_dict["id2label"])
    pred_one_hots: List[List[int]] = [] 
    gt_one_hots: List[List[int]] = []     
    for sample in inf_results:
        pred_one_hot: np.ndarray = np.zeros(label_dim)
        gt_one_hot: np.ndarray = np.zeros(label_dim)
        
        gt_labels: List[str] | str = sample[gt_label_col]
        if isinstance(gt_labels, str):
            gt_labels = [
                x.strip(" ") for x in gt_labels.split(label_splitter)
            ]
        gt_labels = [x for x in gt_labels if x not in {""}]
        if len(gt_labels) == 0:
            continue
        pred_results: List[Tuple[str, float]] = sorted(
            [(k, v) for k, v in sample[pred_results_col].items()],
            reverse=True, 
            key=lambda x: x[1]
        )
        for label in gt_labels:
            label_id: int = label_dict["label2id"][label]
            gt_one_hot[label_id] = 1.0
        for label, score in pred_results:
            if score < min_confidence:
                continue
            label_id: int = label_dict["label2id"][label]
            pred_one_hot[label_id] = 1.0
        if pred_one_hot.sum() == 0 and len(pred_results) > 0:
            top1_label: str = pred_results[0][0]
            top1_label_id: int = label_dict["label2id"][top1_label] 
            pred_one_hot[top1_label_id] = 1.0

        gt_one_hots.append(gt_one_hot.tolist())
        pred_one_hots.append(pred_one_hot.tolist())
    
    verification(train_eval_results_path, gt_one_hots, pred_one_hots)
    results = metrics_func(IntTensor(pred_one_hots), IntTensor(gt_one_hots))
    print(json.dumps(results, indent=2))
    return


if __name__ == "__main__":
    main()

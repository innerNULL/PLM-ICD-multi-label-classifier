# -*- coding: utf-8 -*-
# file: metrics.py
# date: 2023-10-01


import pdb
import torch
from typing import Dict, Optional
from torch import Tensor, IntTensor, FloatTensor


def metrics_func(
    preds_one_hot: IntTensor, label_one_hot: IntTensor, bias: float=1e-6
) -> float:
    # 1 represents correctly predicted positive class
    pred_pos_correctness: IntTensor = preds_one_hot.mul(label_one_hot)
    
    correct_pos_pred_cnt: IntTensor = pred_pos_correctness.sum(dim=1)
    sample_label_cnt: IntTensor = label_one_hot.sum(dim=1) + bias
    pred_label_cnt: IntTensor = preds_one_hot.sum(dim=1) + bias

    macro_recall: FloatTensor = correct_pos_pred_cnt.div(sample_label_cnt).mean()
    macro_precision: FloatTensor = correct_pos_pred_cnt.div(pred_label_cnt).mean()
    macro_f1: FloatTensor = 2 * macro_recall * macro_precision / (macro_recall + macro_precision + bias)

    micro_recall: FloatTensor = correct_pos_pred_cnt.sum() / (sample_label_cnt.sum() + bias)
    micro_precision: FloatTensor = correct_pos_pred_cnt.sum() / (pred_label_cnt.sum() + bias)
    micro_f1: FloatTensor = 2 * micro_recall * micro_precision / (micro_recall + micro_precision + bias)

    return {
        "macro_recall": float(macro_recall.cpu()), 
        "macro_precision": float(macro_precision.cpu()), 
        "macro_f1": float(macro_f1.cpu()), 
        "micro_recall": float(micro_recall.cpu()), 
        "micro_precision": float(micro_precision.cpu()), 
        "micro_f1": float(micro_f1.cpu())
    }


def topk_metrics_func(
    logits: FloatTensor, label_one_hot: IntTensor, top_k: int, bias: float=1e-6
) -> Dict[str, float]:
    out: Dict = {}
    
    probs: FloatTensor = torch.sigmoid(logits)
    topk_thresholds: FloatTensor = probs.topk(top_k, dim=1).values.min(dim=1).values.view(-1, 1)
    output_one_hot: IntTensor = (probs >= topk_thresholds).int()
    metrics: Dict = metrics_func(output_one_hot, label_one_hot, bias)

    out["macro_recall@%i" % top_k] = metrics["macro_recall"]
    out["macro_precision@%i" % top_k] = metrics["macro_precision"]
    out["macro_f1@%i" % top_k] = metrics["macro_f1"]
    out["micro_recall@%i" % top_k] = metrics["micro_recall"]
    out["micro_precision@%i" % top_k] = metrics["micro_precision"]
    out["micro_f1@%i" % top_k] = metrics["micro_f1"]
   
    return out




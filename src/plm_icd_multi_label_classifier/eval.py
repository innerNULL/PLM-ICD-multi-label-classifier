# -*- coding: utf-8 -*-
# file: eval.py
# date: 2025-08-04


import pdb
import json
import torch
import torch.nn.functional as F
from typing import Dict
from torch import device
from torch import LongTensor, FloatTensor, IntTensor
from torch.utils.data import DataLoader 

from .model import PlmMultiLabelEncoder
from .metrics import metrics_func, topk_metrics_func


THRESHOLD: float = 0.6


def evaluation(
    model: PlmMultiLabelEncoder, 
    dataloader: DataLoader, 
    device: device=None, 
    max_sample: int=1e4,
    label_confidence_threshold: float=THRESHOLD,
    verbose: bool=False
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total_cnt: int = 0
    all_logits: List[FloatTensor] = []
    all_label_one_hots: List[FloatTensor] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            curr_label_one_hot: FloatTensor = None
            curr_text_ids: LongTensor = None
            curr_attn_masks: LongTensor = None

            curr_text_ids, curr_attn_masks, curr_label_one_hot = batch

            if device is not None:
                curr_label_one_hot = curr_label_one_hot.to(device)
                curr_text_ids = curr_text_ids.to(device)
                curr_attn_masks = curr_attn_masks.to(device)
            
            curr_logits: FloatTensor = model(curr_text_ids, curr_attn_masks)
            all_logits.append(curr_logits)
            all_label_one_hots.append(curr_label_one_hot)

            total_cnt += curr_text_ids.shape[0]
            if total_cnt >= max_sample:
                break

        logits: FloatTensor = torch.concat(all_logits, dim=0)
        output_label_probs: FloatTensor = torch.sigmoid(logits)
        output_one_hot: FloatTensor = (
            (output_label_probs > label_confidence_threshold).float()
        )
        label_one_hot: FloatTensor = torch.concat(all_label_one_hots, dim=0)
        # Loss
        loss: float = float(
            F.binary_cross_entropy(output_label_probs, label_one_hot).cpu()
        )
        # Metrics
        prob50_metrics: Dict[str, float] = metrics_func(
            output_one_hot.int(), label_one_hot.int()
        )
        #top5_metrics: Dict[str, float] = topk_metrics_func(logits, label_one_hot, top_k=5) 
        #top8_metrics: Dict[str, float] = topk_metrics_func(logits, label_one_hot, top_k=8)
        #top15_metrics: Dict[str, float] = topk_metrics_func(logits, label_one_hot, top_k=15)

        out = {
            "loss": round(loss, 8),  
            "micro_recall": round(prob50_metrics["micro_recall"], 4), 
            "micro_precision": round(prob50_metrics["micro_precision"], 4),
            "micro_f1": round(prob50_metrics["micro_f1"], 4),
            "macro_recall": round(prob50_metrics["macro_recall"], 4), 
            "macro_precision": round(prob50_metrics["macro_precision"], 4),
            "macro_f1": round(prob50_metrics["macro_f1"], 4), 
            #"micro_recall@5": round(top5_metrics["micro_recall@5"], 4), 
            #"micro_precision@5": round(top5_metrics["micro_precision@5"], 4), 
            #"micro_f1@5": round(top5_metrics["micro_f1@5"], 4), 
            #"macro_recall@5": round(top5_metrics["macro_recall@5"], 4), 
            #"macro_precision@5": round(top5_metrics["macro_precision@5"], 4), 
            #"macro_f1@5": round(top5_metrics["macro_f1@5"], 4), 
            #"micro_recall@8": round(top8_metrics["micro_recall@8"], 4), 
            #"micro_precision@8": round(top8_metrics["micro_precision@8"], 4), 
            #"micro_f1@8": round(top8_metrics["micro_f1@8"], 4), 
            #"macro_recall@8": round(top8_metrics["macro_recall@8"], 4), 
            #"macro_precision@8": round(top8_metrics["macro_precision@8"], 4), 
            #"macro_f1@8": round(top8_metrics["macro_f1@8"], 4), 
            #"micro_recall@15": round(top15_metrics["micro_recall@15"], 4), 
            #"micro_precision@15": round(top15_metrics["micro_precision@15"], 4), 
            #"micro_f1@15": round(top15_metrics["micro_f1@15"], 4), 
            #"macro_recall@15": round(top15_metrics["macro_recall@15"], 4), 
            #"macro_precision@15": round(top15_metrics["macro_precision@15"], 4), 
            #"macro_f1@15": round(top15_metrics["macro_f1@15"], 4) 
        }
        if verbose == True:
            out["verbose"] = {}
            out["verbose"]["pred_one_hot"] = output_one_hot.int().tolist()
            out["verbose"]["gt_one_hot"] = label_one_hot.int().tolist()
    return out

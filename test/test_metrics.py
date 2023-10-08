# -*- coding: utf-8 -*-
# file: test_metrics.py
# date: 2023-10-01


import sys, os
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../src")
)

import pdb
from typing import List, Dict
from torch import IntTensor, FloatTensor

from plm_icd_multi_label_classifier.metrics import metrics_func, topk_metrics_func


BIAS: float = 1e-6

# Cases
CASE0_OUTPUT_LOGITS: FloatTensor = FloatTensor(
	[
		[1.15, 6.66, 6.32, 0.52, 6.58, 0.01], 
		[1.11, 1.01, 9.86, 8.88, 9.98, 8.66],
		[0.99, 0.76, 0.16, 0.88, 0.86, 0.01]
	]
)

CASE0_LABEL_ONE_HOT: IntTensor = IntTensor(
	[
		[0, 0, 1, 1, 1, 1],
		[1, 1, 1, 0, 0, 0],
		[0, 1, 1, 0, 1, 0]
	]
)

# top_k == 1 metrics
CASE0_MICRO_RECALL_TOP1: float = (0 + 0 + 0) / (4 + 3 + 3 + BIAS)
CASE0_MICRO_PRECISION_TOP1: float = (0 + 0 + 0) / (1 + 1 + 1 + BIAS)
CASE0_MACRO_RECALL_TOP1: float = (
	0 / (4 + BIAS) + 0 / (3 + BIAS) + 0 / (3 + BIAS)
) / 3
CASE0_MACRO_PRECISION_TOP1: float = (
	0 / (1 + BIAS) + 0 / (1 + BIAS) + 0 / (1 + BIAS)
) / 3

# top_k == 2 metrics
CASE0_MICRO_RECALL_TOP2: float = (1 + 1 + 0) / (4 + 3 + 3 + BIAS)
CASE0_MICRO_PRECISION_TOP2: float = (1 + 1 + 0) / (2 + 2 + 2 + BIAS)
CASE0_MACRO_RECALL_TOP2: float = (
	1 / (4 + BIAS) + 1 / (3 + BIAS) + 0 / (3 + BIAS)
) / 3
CASE0_MACRO_PRECISION_TOP2: float = (
	1 / (2 + BIAS) + 1 / (2 + BIAS) + 0 / (2 + BIAS)
) / 3

# top_k == 4 metrics
CASE0_MICRO_RECALL_TOP4: float = (2 + 1 + 2) / (4 + 3 + 3 + BIAS)
CASE0_MICRO_PRECISION_TOP4: float = (2 + 1 + 2) / (4 * 3 + BIAS)
CASE0_MACRO_RECALL_TOP4: float = (
    2 / (4 + BIAS) + 1 / (3 + BIAS) + 2 / (3 + BIAS)
) / 3
CASE0_MACRO_PRECISION_TOP4: float = (
    2 / (4 + BIAS) + 1 / (4 + BIAS) + 2 / (4 + BIAS)
) / 3


def test_metric_func_0() -> None:
    output_one_hot: List[List[int]] = [[0, 1, 1, 0, 1, 0]]
    label_one_hot: List[List[int]] = [[0, 0, 1, 1, 1, 1]]
    
    micro_recall: float = 2 / (4 + BIAS)
    micro_precision: float = 2 / (3 + BIAS)
    macro_recall: float = (2 / (4 + BIAS)) / 1
    macro_precision: float = (2 / (3 + BIAS)) / 1

    cal_metrics: Dict[str, float] = metrics_func(
        IntTensor(output_one_hot), IntTensor(label_one_hot), BIAS
    )

    assert(round(cal_metrics["micro_recall"], 2) == round(micro_recall, 2))
    assert(round(cal_metrics["micro_precision"], 2) == round(micro_precision, 2))
    assert(round(cal_metrics["macro_recall"], 2) == round(macro_recall, 2))
    assert(round(cal_metrics["macro_precision"], 2) == round(macro_precision, 2))


def test_metric_func_1() -> None:
    output_one_hot: List[List[int]] = [
        [0, 1, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 1]
    ]
    label_one_hot: List[List[int]] = [
        [0, 0, 1, 1, 1, 1], 
        [1, 1, 1, 0, 0, 0], 
        [0, 1, 1, 0, 1, 0]
    ]
    
    micro_recall: float = (2 + 1 + 1) / (10 + BIAS)
    micro_precision: float = (2 + 1 + 1) / (11 + BIAS)
    macro_recall: float = (
        2 / (4 + BIAS) + 1 / (3 + BIAS) + 1 / (3 + BIAS)
    ) / 3
    macro_precision: float = (
        2 / (3 + BIAS) + 1 / (4 + BIAS) + 1 / (4 + BIAS)
    ) / 3

    cal_metrics: Dict[str, float] = metrics_func(
        IntTensor(output_one_hot), IntTensor(label_one_hot), BIAS
    )
    
    assert(round(cal_metrics["micro_recall"], 2) == round(micro_recall, 2))
    assert(round(cal_metrics["micro_precision"], 2) == round(micro_precision, 2))
    assert(round(cal_metrics["macro_recall"], 2) == round(macro_recall, 2))
    assert(round(cal_metrics["macro_precision"], 2) == round(macro_precision, 2))


def test_topk_metrics_func_0() -> None:
    top_ks: List[int] = [1, 2, 4]
    for top_k in top_ks:
        out_metrics: Dict = topk_metrics_func(
            CASE0_OUTPUT_LOGITS, CASE0_LABEL_ONE_HOT, top_k=top_k, bias=BIAS
        )
        
        if top_k == 1:
            CASE0_MICRO_RECALL_TOP1 == out_metrics["micro_recall@1"]
            CASE0_MICRO_PRECISION_TOP1 == out_metrics["micro_precision@1"]
            CASE0_MACRO_RECALL_TOP1 == out_metrics["macro_recall@1"]
            CASE0_MACRO_PRECISION_TOP1 == out_metrics["macro_precision@1"]
        elif top_k == 2:
            CASE0_MICRO_RECALL_TOP2 == out_metrics["micro_recall@2"]
            CASE0_MICRO_PRECISION_TOP2 == out_metrics["micro_precision@2"]
            CASE0_MACRO_RECALL_TOP2 == out_metrics["macro_recall@2"]
            CASE0_MACRO_PRECISION_TOP2 == out_metrics["macro_precision@2"]
        elif top_k == 4:
            CASE0_MICRO_RECALL_TOP4 = out_metrics["micro_recall@4"]
            CASE0_MICRO_PRECISION_TOP4 = out_metrics["micro_precision@4"]
            CASE0_MACRO_RECALL_TOP4 = out_metrics["macro_recall@4"]
            CASE0_MACRO_PRECISION_TOP4 = out_metrics["macro_precision@4"]
        else:
            raise Exception("No case provided")
           


# -*- coding: utf-8 -*-
# file: test_metrics.py
# date: 2023-10-01


import sys, os
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)

import pdb
from typing import List, Dict
from torch import IntTensor, FloatTensor

from src.metrics import metrics_func


BIAS: float=1e-6

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


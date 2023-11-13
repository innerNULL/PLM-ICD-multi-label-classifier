# -*- coding: utf-8 -*-
# file: test_loss.py
# date: 2023-10-07


import sys, os
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../src")
)

import pdb
import torch
from typing import List, Dict
from torch import IntTensor, FloatTensor

from plm_icd_multi_label_classifier.loss import focal_loss_with_logits


def test_focal_loss_with_logits_0() -> None:
    bias: float = 1e-10
    gamma: float = 2
    label_one_hot: IntTensor = IntTensor([[1, 0, 1, 0]])
    logits: FloatTensor = FloatTensor([[10.0, 3.0, 4.0, 1.0]])
    probs: FloatTensor = torch.sigmoid(logits) + bias
    loss_output: FloatTensor = focal_loss_with_logits(
        logits, label_one_hot, 
        bias=bias, gamma=gamma
    )
    loss_target: FloatTensor = -1 * (
        torch.pow(1 - probs[0, 0], gamma) * torch.log(probs[0, 0]) 
        + torch.pow(probs[0, 1], gamma) * torch.log(1 - probs[0, 1]) 
        + torch.pow(1 - probs[0, 2], gamma) * torch.log(probs[0, 2]) 
        + torch.pow(probs[0, 3], gamma) * torch.log(1 - probs[0, 3])
    ) / 4
    assert(loss_output == loss_target)


def test_focal_loss_with_logits_1() -> None:
    bias: float = 1e-10
    gamma: float = 2
    label_one_hot: IntTensor = IntTensor([[1, 0, 1, 0], [1, 0, 1, 0]])
    logits: FloatTensor = FloatTensor([
        [10.0, 3.0, 4.0, 1.0], [10.0, 3.0, 4.0, 1.0]
    ])
    probs: FloatTensor = torch.sigmoid(logits) + bias
    loss_output: FloatTensor = focal_loss_with_logits(
        logits, label_one_hot, 
        bias=bias, gamma=gamma
    )
    loss_target: FloatTensor = -1 * (
        torch.pow(1 - probs[0, 0], gamma) * torch.log(probs[0, 0]) 
        + torch.pow(probs[0, 1], gamma) * torch.log(1 - probs[0, 1]) 
        + torch.pow(1 - probs[0, 2], gamma) * torch.log(probs[0, 2]) 
        + torch.pow(probs[0, 3], gamma) * torch.log(1 - probs[0, 3])
    ) / 4 * 2 / 2
    assert(loss_output == loss_target)

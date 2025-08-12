# -*- coding: utf-8 -*-
# file: loss.py
# date: 2023-10-07


import pdb
import torch
import torch.nn.functional as F
from typing import Callable, Optional, Dict
from torch import LongTensor, FloatTensor, IntTensor


def bce_with_logits(
    logits: FloatTensor, label_one_hot: FloatTensor, bias: float=1e-10
) -> FloatTensor:
    label_probs: FloatTensor = torch.sigmoid(logits) + bias
    bin_cross_entropies: FloatTensor = \
        label_one_hot.mul(torch.log(label_probs)) \
        + (1 - label_one_hot).mul(torch.log(1 - label_probs))
    loss: FloatTensor = -bin_cross_entropies.mean(dim=1)
    return loss.mean()


def focal_loss_with_logits(
    logits: FloatTensor, 
    label_one_hot: FloatTensor, 
    bias: float=1e-10, 
    gamma: float=2
) -> FloatTensor:
    """
    Refer to Focal Loss for Dense Object Detection
    https://arxiv.org/pdf/1708.02002

    This's still under testing
    """
    pred_probs_pos: FloatTensor = torch.sigmoid(logits) * label_one_hot
    pred_probs_neg: FloatTensor = (1 - torch.sigmoid(logits)) * (1 - label_one_hot) 
    p_t: FloatTensor = pred_probs_pos + pred_probs_neg
    modulating_factor: FloatTensor = torch.pow(1 - p_t, gamma)
    loss: FloatTensor = -1 * modulating_factor * torch.log(p_t)
    return loss.mean(dim=1).mean()


def loss_factory(configs: Dict) -> Optional[Callable]:
    name: str = configs["name"]
    loss_fn: Optional[Callable] = None
    if name in {"bce", "binary_cross_entropy"}:
        def _loss_fn(
            logits: FloatTensor, 
            label_one_hot: IntTensor
        ) -> FloatTensor:
            return F.binary_cross_entropy(torch.sigmoid(logits), label_one_hot)
        loss_fn = _loss_fn
    elif name in {"focal_loss"}:
        gamma: float = configs["gamma"]
        def _loss_fn(
            logits: FloatTensor, 
            label_one_hot: IntTensor
        ) -> FloatTensor:
            return focal_loss_with_logits(
                logits, 
                label_one_hot, 
                gamma=gamma
            )
        loss_fn = _loss_fn
    return loss_fn

# -*- coding: utf-8 -*-
# file: loss.py
# date: 2023-10-07


import pdb
import torch
from torch import FloatTensor


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
    logits: FloatTensor, label_one_hot: FloatTensor, bias: float=1e-10, gamma: float=2
) -> FloatTensor:
    """
    I got this from this paper
    "Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution", 
    but seems it at least not work for all medical dataset I had.
    """
    label_probs: FloatTensor = torch.sigmoid(logits) + bias
    loss: FloatTensor = \
        label_one_hot.mul(torch.log(label_probs)).mul(torch.pow(1 - label_probs, gamma)) \
        + (1 - label_one_hot).mul(torch.log(1 - label_probs)).mul(torch.pow(label_probs, gamma))
    return (-1 * loss).mean(dim=1).mean()

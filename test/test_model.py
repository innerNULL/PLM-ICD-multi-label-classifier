# -*- coding: utf-8 -*-
# file: test_model.py
# date: 2023-09-23


import sys, os
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)

import pdb
from typing import List
from transformers import AutoTokenizer

from src import text
from src.model import PlmMultiLabelEncoder 


CHUNK_SIZES: List[int] = [512]
CHUNK_NUMS: List[int] = [2]
LABEL_NUMS: List[int] = [1000]
HF_LM: str = "distilbert-base-uncased"
LM_EMBD_DIM: int = 768


def test_PlmMultiLabelEncoder_forward() -> None:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(HF_LM)
    
    assert(len(CHUNK_SIZES) == len(CHUNK_NUMS))
    assert(len(CHUNK_SIZES) == len(LABEL_NUMS))

    for i in range(len(CHUNK_SIZES)):
        chunk_size: int = CHUNK_SIZES[i]
        chunk_num: int = CHUNK_NUMS[i]
        label_num: int = LABEL_NUMS[i]
        model: PlmMultiLabelEncoder = PlmMultiLabelEncoder(
            label_num, HF_LM, LM_EMBD_DIM, chunk_size, chunk_num
        )

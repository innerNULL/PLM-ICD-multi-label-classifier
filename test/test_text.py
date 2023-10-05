# -*- coding: utf-8 -*-
# file: test_text.py
# date: 2023-09-22


import sys, os
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)

import pdb
from transformers import AutoTokenizer

from src import text


HF_LM: str = "distilbert-base-uncased"


def test_tokenize_to_chunks() -> None:
    base_fake_text: str = "This is a fake text description"
    chunk_size: int = 512
    chunk_num: int = 2
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(HF_LM)
    padding_idx: int = tokenizer(
        tokenizer.pad_token, padding="longest", truncation=True
    )["input_ids"][1]

    for i in [1, 5, 10, 50, 100, 500, 1000, 5000]:
        fake_text: str = ".".join([base_fake_text] * i)
        text_id_chunks: List[List[int]] = []
        attn_mask_chunks: List[List[int]] = []
        text_id_chunks, attn_mask_chunks = text.tokenize_to_chunks(
            fake_text, tokenizer, chunk_size, chunk_num
        )
        
        assert(len(text_id_chunks) == chunk_num)
        assert(len(attn_mask_chunks) == len(text_id_chunks))
        
        for i in range(len(text_id_chunks)):
            text_id_chunk: List[int] = text_id_chunks[i]
            attn_mask_chunk: List[int] = attn_mask_chunks[i]
            
            assert(len(text_id_chunk) == len(attn_mask_chunk))
            assert(
                len([x for x in text_id_chunk if x != padding_idx]) 
                == len([x for x in attn_mask_chunk if x != 0])
            )
            
            max_non_pad_idx: int = max(
                [i for i in range(len(text_id_chunk)) if text_id_chunk[i] != padding_idx] 
                + [-1]
            )
            min_pad_idx: int = min(
                [i for i in range(len(text_id_chunk)) if text_id_chunk[i] == padding_idx] 
                + [sys.maxsize]
            )
            assert(max_non_pad_idx < min_pad_idx)

            max_non_mask_idx: int = max(
                [i for i in range(len(attn_mask_chunk)) if attn_mask_chunk[i] != 0] 
                + [-1]
            )
            min_mask_idx: int = min(
                [i for i in range(len(attn_mask_chunk)) if attn_mask_chunk[i] == 0]
                + [sys.maxsize]
            )
            assert(max_non_mask_idx < min_mask_idx)


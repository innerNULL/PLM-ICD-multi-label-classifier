# -*- coding: utf-8 -*-
# file: text.py
# date: 2023-09-22


import pdb
from typing import Tuple, List
from transformers import AutoTokenizer


def token_ids_chunking(
    text_ids: List[int], attn_marks: List[int], 
    chunk_size: int, chunk_num: int, padding_idx: int, check: bool=True
) -> Tuple[ List[List[int]], List[List[int]] ]:
    if check:
        assert(len(text_ids) == len(attn_marks))

    total_token_num: int = chunk_num * chunk_size
    text_ids = text_ids + [padding_idx] * (total_token_num - len(text_ids))
    attn_marks = attn_marks + [0] * (total_token_num - len(attn_marks))

    if check:
        assert(len(text_ids) == len(attn_marks))

    token_id_chunks: List[List[int]] = []
    attn_mark_chunks: List[List[int]] = []
    for i in range(chunk_num):
        start_idx: int = i * chunk_size
        end_idx: int = (i + 1) * chunk_size
        curr_chunk_token_ids: List[int] = text_ids[start_idx:end_idx] 
        curr_chunk_attn_marks: List[int] = attn_marks[start_idx:end_idx]
        token_id_chunks.append(curr_chunk_token_ids)
        attn_mark_chunks.append(curr_chunk_attn_marks)

    return (token_id_chunks, attn_mark_chunks)


def tokenize_to_chunks(
    text, tokenizer: AutoTokenizer, chunk_size: int, chunk_num: int
) -> Tuple[ List[List[int]], List[List[int]] ]:
    tokenize_results: Dict[str, List[int]] = tokenizer(
        text, padding=False, truncation=False
    )
    padding_idx: int = tokenizer(
        tokenizer.pad_token, padding="longest", truncation=True
    )["input_ids"][1]
    text_ids: List[int] = tokenize_results["input_ids"]
    attn_masks: List[int] = tokenize_results["attention_mask"]
    
    return token_ids_chunking(
        text_ids, attn_masks, chunk_size, chunk_num, padding_idx, False
    )


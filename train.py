# -*- coding: utf-8 -*-
# file: train.py
# date: 2023-09-22


import pdb
import os
import json
import torch
from typing import Dict
from transformers import AutoTokenizer
from torch import device
from torch import LongTensor, FloatTensor
from torch.utils.data import DataLoader 
from torch.optim import AdamW

from src import text
from src.model import PlmMultiLabelEncoder
from src.data import TextOnlyDataset


CHUNK_SIZE: int = 512
CHUNK_NUM: int = 2
HF_LM: str = "distilbert-base-uncased"
DATA_DIR: str = "_data/etl/mimic3"


def loss_fn(logits: FloatTensor, label_one_hot: FloatTensor) -> FloatTensor:
    label_probs: FloatTensor = torch.sigmoid(logits)
    bin_cross_entropies: FloatTensor = \
        label_one_hot.mul(torch.log(label_probs)) \
        + (1 - label_one_hot).mul(torch.log(1 - label_probs))
    loss: FloatTensor = -bin_cross_entropies.mean(dim=1)
    return loss.mean()


if __name__ == "__main__":
    device: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dict: Dict = json.loads(
        open(os.path.join(DATA_DIR, "dict.json"), "r").read()
    )
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(HF_LM)
    model: PlmMultiLabelEncoder = PlmMultiLabelEncoder(
        len(data_dict["label2id"]), HF_LM, 768, CHUNK_SIZE, CHUNK_NUM
    )
    dataset: TextOnlyDataset = TextOnlyDataset(
        os.path.join(DATA_DIR, "train.csv"), 
        os.path.join(DATA_DIR, "dict.json"), tokenizer, "text"
    )
    dataloader: DataLoader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer: AdamW = AdamW(model.parameters(), lr=1e-3)
    
    model.to(device)

    for epoch in range(3):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            label_one_hot: FloatTensor = None
            text_ids: LongTensor = None
            attn_masks: LongTensor = None

            text_ids, attn_masks, label_one_hot = batch

            label_one_hot = label_one_hot.to(device)
            text_ids = text_ids.to(device)
            attn_masks = attn_masks.to(device)

            model.train()
            logits: FloatTensor = model(text_ids, attn_masks)
            loss: FloatTensor = loss_fn(logits, label_one_hot)
            
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print("loss=%f" % loss)
            
            model.eval()
        

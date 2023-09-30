# -*- coding: utf-8 -*-
# file: train.py
# date: 2023-09-22


import pdb
import os
import json
import torch
import torch.nn.functional as F
from typing import Dict
from transformers import AutoTokenizer
from torch import device
from torch import LongTensor, FloatTensor
from torch.utils.data import DataLoader 
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from src import text
from src.model import PlmMultiLabelEncoder
from src.data import TextOnlyDataset


CHUNK_SIZE: int = 128
CHUNK_NUM: int = 24
#HF_LM: str = "dmis-lab/biobert-base-cased-v1.1"
HF_LM: str = "distilbert-base-uncased"
DATA_DIR: str = "_data/etl/mimic3"


def loss_fn(
    logits: FloatTensor, label_one_hot: FloatTensor, bias: float=1e-10
) -> FloatTensor:
    label_probs: FloatTensor = torch.sigmoid(logits) + bias
    bin_cross_entropies: FloatTensor = \
        label_one_hot.mul(torch.log(label_probs)) \
        + (1 - label_one_hot).mul(torch.log(1 - label_probs))
    loss: FloatTensor = -bin_cross_entropies.mean(dim=1)
    return loss.mean()


def eval(
    model: PlmMultiLabelEncoder, dataloader: DataLoader, device: device, 
    max_sample: int=1e4
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total_cnt: int = 0
    all_loss: List[float] = []
    all_micro_recall: List[float] = []
    all_micro_precision: List[float] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            label_one_hot: FloatTensor = None
            text_ids: LongTensor = None
            attn_masks: LongTensor = None

            text_ids, attn_masks, label_one_hot = batch

            label_one_hot = label_one_hot.to(device)
            text_ids = text_ids.to(device)
            attn_masks = attn_masks.to(device)
            
            logits: FloatTensor = model(text_ids, attn_masks)
            output_label_probs: FloatTensor = torch.sigmoid(logits)

            loss: float = float(
                F.binary_cross_entropy(output_label_probs, label_one_hot).cpu()
            )
            all_loss.append(loss)

            total_cnt += text_ids.shape[0]
            if total_cnt >= max_sample:
                break

    return {
        "loss": sum(all_loss) / len(all_loss)
    }

if __name__ == "__main__":
    torch.manual_seed(32)

    device: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dict: Dict = json.loads(
        open(os.path.join(DATA_DIR, "dict.json"), "r").read()
    )
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(HF_LM)
    model: PlmMultiLabelEncoder = PlmMultiLabelEncoder(
        len(data_dict["label2id"]), HF_LM, 768, CHUNK_SIZE, CHUNK_NUM
    )
    
    train_dataset: TextOnlyDataset = TextOnlyDataset(
        os.path.join(DATA_DIR, "train.csv"), 
        os.path.join(DATA_DIR, "dict.json"), tokenizer, "text", 
        chunk_size=CHUNK_SIZE, chunk_num=CHUNK_NUM
    )
    dev_dataset: TextOnlyDataset = TextOnlyDataset(
        os.path.join(DATA_DIR, "dev.csv"), 
        os.path.join(DATA_DIR, "dict.json"), tokenizer, "text", 
        chunk_size=CHUNK_SIZE, chunk_num=CHUNK_NUM
    )
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    dev_dataloader: DataLoader = DataLoader(dev_dataset, batch_size=8, shuffle=True)
    
    optimizer: AdamW = AdamW(model.parameters(), lr=5e-5)
    scheduler = LinearLR(optimizer, total_iters=2000)
    
    model.to(device)
   
    global_step_id: int = 0
    for epoch_id, epoch in enumerate(range(3)):
        for batch_id, batch in enumerate(train_dataloader):
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
            #loss: FloatTensor = loss_fn(logits, label_one_hot)
            loss: FloatTensor = F.binary_cross_entropy(torch.sigmoid(logits), label_one_hot)
            
            loss.backward()
            optimizer.step()
            
            if batch_id % 10 == 0:
                print("loss=%f" % loss)
            if batch_id % 100  == 0:
                print(eval(model, dev_dataloader, device, 1000)) 
            
            model.eval()
        

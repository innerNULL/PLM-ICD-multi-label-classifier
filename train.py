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
from torch import LongTensor, FloatTensor, IntTensor
from torch.utils.data import DataLoader 
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from src import text
from src.model import PlmMultiLabelEncoder
from src.data import TextOnlyDataset
from src.metrics import metrics_func


CONF: Dict = {
    "chunk_size": 512, 
    "chunk_num": 3, 
    "hf_lm": "distilbert-base-uncased",
    "lm_hidden_dim": 768,
    "data_dir": "_data/etl/mimic3"
}


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
    all_micro_f1: List[float] = []
    all_macro_recall: List[float] = []
    all_macro_precision: List[float] = []
    all_macro_f1: List[float] = []

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
            output_one_hot: IntTensor = (output_label_probs > 0.5).int()

            # Loss
            loss: float = float(
                F.binary_cross_entropy(output_label_probs, label_one_hot).cpu()
            )
            all_loss.append(loss)
            
            # Metrics
            curr_metrics: Dict[str, float] = metrics_func(
                output_one_hot.int(), label_one_hot.int()
            )
            all_micro_recall.append(curr_metrics["micro_recall"])
            all_micro_precision.append(curr_metrics["micro_precision"])
            all_micro_f1.append(curr_metrics["micro_f1"])
            all_macro_recall.append(curr_metrics["macro_recall"])
            all_macro_precision.append(curr_metrics["macro_precision"])
            all_macro_f1.append(curr_metrics["macro_f1"])

            total_cnt += text_ids.shape[0]
            if total_cnt >= max_sample:
                break

    return {
        "loss": round(sum(all_loss) / len(all_loss), 8),  
        "micro_recall": round(sum(all_micro_recall) / len(all_micro_recall), 4), 
        "micro_precision": round(sum(all_micro_precision) / len(all_micro_precision), 4),
        "micro_f1": round(sum(all_micro_f1) / len(all_micro_f1), 4),
        "macro_recall": round(sum(all_macro_recall) / len(all_macro_recall), 4), 
        "macro_precision": round(sum(all_macro_precision) / len(all_macro_precision), 4),
        "macro_f1": round(sum(all_macro_f1) / len(all_macro_f1), 4)
    }


def train_func(configs: Dict) -> None:
    device: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dict_path: str = os.path.join(configs["data_dir"], "dict.json")
    train_data_path: str = os.path.join(configs["data_dir"], "train.csv")
    dev_data_path: str = os.path.join(configs["data_dir"], "dev.csv")

    data_dict: Dict = json.loads(open(data_dict_path, "r").read())
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(configs["hf_lm"])
    model: PlmMultiLabelEncoder = PlmMultiLabelEncoder(
        len(data_dict["label2id"]), 
        configs["hf_lm"], configs["lm_hidden_dim"], configs["chunk_size"], configs["chunk_num"]
    )
    
    train_dataset: TextOnlyDataset = TextOnlyDataset(
        train_data_path, data_dict_path, tokenizer, "text", 
        chunk_size=configs["chunk_size"], chunk_num=configs["chunk_num"]
    )
    dev_dataset: TextOnlyDataset = TextOnlyDataset(
        dev_data_path, data_dict_path, tokenizer, "text", 
        chunk_size=configs["chunk_size"], chunk_num=configs["chunk_num"]
    )
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    dev_dataloader: DataLoader = DataLoader(dev_dataset, batch_size=64, shuffle=True)
    
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
           
            model.eval()
            if batch_id % 10 == 0:
                print("loss=%f" % loss)
            if batch_id % 500  == 0:
                print(eval(model, dev_dataloader, device, 1000)) 
       
if __name__ == "__main__":
    torch.manual_seed(32)
    train_func(CONF)

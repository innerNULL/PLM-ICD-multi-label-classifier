# -*- coding: utf-8 -*-
# file: train.py
# date: 2023-09-22


import pdb
import sys
import os
import tempfile
import json
import torch
import ray.train
import torch.nn.functional as F
from typing import Dict
from transformers import AutoTokenizer
from torch import device
from torch import LongTensor, FloatTensor, IntTensor
from torch.utils.data import DataLoader 
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from src import text
from src.model import PlmMultiLabelEncoder
from src.data import TextOnlyDataset
from src.metrics import metrics_func, topk_metrics_func


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
    model: PlmMultiLabelEncoder, dataloader: DataLoader, device: device=None, 
    max_sample: int=1e4
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total_cnt: int = 0
    all_logits: List[FloatTensor] = []
    all_label_one_hots: List[FloatTensor] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            curr_label_one_hot: FloatTensor = None
            curr_text_ids: LongTensor = None
            curr_attn_masks: LongTensor = None

            curr_text_ids, curr_attn_masks, curr_label_one_hot = batch

            if device is not None:
                curr_label_one_hot = curr_label_one_hot.to(device)
                curr_text_ids = curr_text_ids.to(device)
                curr_attn_masks = curr_attn_masks.to(device)
            
            curr_logits: FloatTensor = model(curr_text_ids, curr_attn_masks)
            all_logits.append(curr_logits)
            all_label_one_hots.append(curr_label_one_hot)

            total_cnt += curr_text_ids.shape[0]
            if total_cnt >= max_sample:
                break

        logits: FloatTensor = torch.concat(all_logits, dim=0)
        output_label_probs: FloatTensor = torch.sigmoid(logits)
        output_one_hot: FloatTensor = (output_label_probs > 0.5).float()
        label_one_hot: FloatTensor = torch.concat(all_label_one_hots, dim=0)
        # Loss
        loss: float = float(
            F.binary_cross_entropy(output_label_probs, label_one_hot).cpu()
        )
        # Metrics
        prob50_metrics: Dict[str, float] = metrics_func(
            output_one_hot.int(), label_one_hot.int()
        )
        top5_metrics: Dict[str, float] = topk_metrics_func(logits, label_one_hot, top_k=5) 
        top8_metrics: Dict[str, float] = topk_metrics_func(logits, label_one_hot, top_k=8)
        top15_metrics: Dict[str, float] = topk_metrics_func(logits, label_one_hot, top_k=15)

        out = {
            "loss": round(loss, 8),  
            "micro_recall": round(prob50_metrics["micro_recall"], 4), 
            "micro_precision": round(prob50_metrics["micro_precision"], 4),
            "micro_f1": round(prob50_metrics["micro_f1"], 4),
            "macro_recall": round(prob50_metrics["macro_recall"], 4), 
            "macro_precision": round(prob50_metrics["macro_precision"], 4),
            "macro_f1": round(prob50_metrics["macro_f1"], 4), 
            "micro_recall@5": round(top5_metrics["micro_recall@5"], 4), 
            "micro_precision@5": round(top5_metrics["micro_precision@5"], 4), 
            "micro_f1@5": round(top5_metrics["micro_f1@5"], 4), 
            "macro_recall@5": round(top5_metrics["macro_recall@5"], 4), 
            "macro_precision@5": round(top5_metrics["macro_precision@5"], 4), 
            "macro_f1@5": round(top5_metrics["macro_f1@5"], 4), 
            "micro_recall@8": round(top8_metrics["micro_recall@8"], 4), 
            "micro_precision@8": round(top8_metrics["micro_precision@8"], 4), 
            "micro_f1@8": round(top8_metrics["micro_f1@8"], 4), 
            "macro_recall@8": round(top8_metrics["macro_recall@8"], 4), 
            "macro_precision@8": round(top8_metrics["macro_precision@8"], 4), 
            "macro_f1@8": round(top8_metrics["macro_f1@8"], 4), 
            "micro_recall@15": round(top15_metrics["micro_recall@15"], 4), 
            "micro_precision@15": round(top15_metrics["micro_precision@15"], 4), 
            "micro_f1@15": round(top15_metrics["micro_f1@15"], 4), 
            "macro_recall@15": round(top15_metrics["macro_recall@15"], 4), 
            "macro_precision@15": round(top15_metrics["macro_precision@15"], 4), 
            "macro_f1@15": round(top15_metrics["macro_f1@15"], 4) 
        }
    return out

def train_func(configs: Dict) -> None:
    torch.manual_seed(configs["random_seed"])

    device: device = None
    if configs["training_engine"] == "torch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        train_data_path, data_dict_path, tokenizer, 
        text_col=configs["text_col"], label_col=configs["label_col"],
        chunk_size=configs["chunk_size"], chunk_num=configs["chunk_num"]
    )
    dev_dataset: TextOnlyDataset = TextOnlyDataset(
        dev_data_path, data_dict_path, tokenizer,
        text_col=configs["text_col"], label_col=configs["label_col"],
        chunk_size=configs["chunk_size"], chunk_num=configs["chunk_num"]
    )
    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=configs["single_worker_batch_size"], shuffle=True
    )
    dev_dataloader: DataLoader = DataLoader(dev_dataset, batch_size=64, shuffle=True)
    
    optimizer: AdamW = AdamW(model.parameters(), lr=configs["lr"])
    scheduler = LinearLR(optimizer, total_iters=2000)
   
    if configs["training_engine"] == "torch":
        model.to(device)
    elif configs["training_engine"] == "ray":
        model = ray.train.torch.prepare_model(model)
        train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
        dev_dataloader = ray.train.torch.prepare_data_loader(dev_dataloader)
   
    global_step_id: int = 0
    for epoch_id, epoch in enumerate(range(configs["epochs"])):
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
            if batch_id % 10 == 0 and configs["training_engine"] == "torch":
                print("loss=%f" % loss)
            if batch_id % configs["log_period"]  == 0:
                eval_metrics: Dict[str, float] = eval(
                    model, dev_dataloader, device, configs["single_worker_eval_size"]
                )
                eval_metrics["train_loss"] = round(float(loss.detach().cpu()), 6)
                eval_metrics["epoch"] = epoch_id
                eval_metrics["batch"] = batch_id
                eval_metrics["step"] = global_step_id
                if configs["training_engine"] == "torch":
                    print(eval_metrics)
                elif configs["training_engine"] == "ray":
                    ray.train.report(metrics=eval_metrics)

            if global_step_id % configs["dump_period"]  == 0:
                version: str = "step{}-batch{}-epoch{}".format(global_step_id, batch_id, epoch_id)
                ckpt_dir: str = os.path.join(configs["ckpt_dir"], version)
                os.system("mkdir -p %s" % ckpt_dir)
                print("Saving ckpt to %s" % ckpt_dir)
                if configs["training_engine"] == "torch":
                    open(os.path.join(ckpt_dir, "train.json"), "w").write(json.dumps(configs))
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
                elif configs["training_engine"] == "ray":
                    if ray.train.get_context().get_world_rank() == 0:
                        open(os.path.join(ckpt_dir, "train.json"), "w").write(json.dumps(configs))
                        torch.save(model.module.state_dict(), os.path.join(ckpt_dir, "model.pt"))

            global_step_id += 1

    final_ckpt_dir: str = os.path.join(configs["ckpt_dir"], "final")
    print("Saving final ckpt to %s" % final_ckpt_dir)
    if configs["training_engine"] == "torch":
        open(os.path.join(final_ckpt_dir, "train.json"), "w").write(json.dumps(configs))
        torch.save(model.state_dict(), os.path.join(final_ckpt_dir, "model.pt"))
    elif configs["training_engine"] == "ray":
        if ray.train.get_context().get_world_rank() == 0:
            open(os.path.join(final_ckpt_dir, "train.json"), "w").write(json.dumps(configs))
            torch.save(model.module.state_dict(), os.path.join(final_ckpt_dir, "model.pt"))


if __name__ == "__main__":
    train_conf: Dict = json.loads(open(sys.argv[1], "r").read())
    train_conf["data_dir"] = os.path.abspath(train_conf["data_dir"])
    train_conf["ckpt_dir"] = os.path.abspath(train_conf["ckpt_dir"])
    print("Training config:\n{}".format(train_conf))

    if train_conf["training_engine"] == "torch":
        train_func(train_conf)
    elif train_conf["training_engine"] == "ray":
        scaling_config = ScalingConfig(
            num_workers=train_conf["workers"], use_gpu=(train_conf["gpu"] == "true")
        )
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=train_conf,
            scaling_config=scaling_config,
        )
        result = trainer.fit()
        print(f"Training result: {result}")

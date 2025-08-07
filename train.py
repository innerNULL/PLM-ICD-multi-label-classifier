# -*- coding: utf-8 -*-
# file: train.py
# date: 2023-09-22


import pdb
import sys
import os
import tempfile
import json
import torch
import random
import ray.train
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from transformers import AutoTokenizer
from torch import device
from torch import LongTensor, FloatTensor, IntTensor
from torch.utils.data import DataLoader 
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from src.plm_icd_multi_label_classifier import text
from src.plm_icd_multi_label_classifier.model import PlmMultiLabelEncoder
from src.plm_icd_multi_label_classifier.data import TextOnlyDataset
from src.plm_icd_multi_label_classifier.metrics import metrics_func, topk_metrics_func
from src.plm_icd_multi_label_classifier.eval import evaluation


def get_lr(optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group['lr']


def init_with_ckpt(net: PlmMultiLabelEncoder, ckpt_root_path: str, engine: str) -> None:
    ckpts: List[str] = [x for x in os.listdir(ckpt_root_path) if x != "bak"]
    if len(ckpts) == 0:
        print("No existing CKPT")
        return 
    ckpts = sorted(
        ckpts, 
        key=lambda x: int(x.split("-")[0].replace("step", "")), reverse=True
    )
    ckpt_path = os.path.join(ckpt_root_path, ckpts[0], "model.pt")

    if engine == "torch":
        net.load_state_dict(torch.load(ckpt_path))
    elif engine == "ray":
        net.module.load_state_dict(torch.load(ckpt_path))
    print("Finished loading CKPT from %s" % ckpt_path)
    print(
        "Please remember to remove original CKPT '%s' manually" 
        % os.path.join(ckpt_root_path, ckpts[0])
    )


def ckpt_dump(
    model: torch.nn.Module,
    global_step_id: int,
    batch_id: int, 
    epoch_id: int,
    configs: Dict,
    ckpt_dir: Optional[str]=None
) -> None:
    if ckpt_dir is None:
        version: str = "step{}-batch{}-epoch{}".format(global_step_id, batch_id, epoch_id)
        ckpt_dir = os.path.join(configs["ckpt_dir"], version)
    os.system("mkdir -p %s" % ckpt_dir)
    print("Saving ckpt to %s" % ckpt_dir)
    if configs["training_engine"] == "torch":
        open(os.path.join(ckpt_dir, "train.json"), "w").write(json.dumps(configs))
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    elif configs["training_engine"] == "ray":
        if ray.train.get_context().get_world_rank() == 0:
            open(os.path.join(ckpt_dir, "train.json"), "w").write(json.dumps(configs))
            try:
                torch.save(model.module.state_dict(), os.path.join(ckpt_dir, "model.pt"))
            except:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))


def loss_fn(
    logits: FloatTensor, label_one_hot: FloatTensor, bias: float=1e-10
) -> FloatTensor:
    label_probs: FloatTensor = torch.sigmoid(logits) + bias
    bin_cross_entropies: FloatTensor = \
        label_one_hot.mul(torch.log(label_probs)) \
        + (1 - label_one_hot).mul(torch.log(1 - label_probs))
    loss: FloatTensor = -bin_cross_entropies.mean(dim=1)
    return loss.mean()


def train_func(configs: Dict) -> None:
    torch.manual_seed(configs["random_seed"])
    random.seed(configs["random_seed"])
    np.random.seed(configs["random_seed"])

    device: device = None
    if configs["training_engine"] == "torch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dict_path: str = os.path.join(configs["data_dir"], "dict.json")
    train_data_path: str = os.path.join(configs["data_dir"], "train.jsonl")
    dev_data_path: str = os.path.join(configs["data_dir"], "dev.jsonl")

    data_dict: Dict = json.loads(open(data_dict_path, "r").read())
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(configs["hf_lm"])
    model: PlmMultiLabelEncoder = PlmMultiLabelEncoder(
        len(data_dict["label2id"]), 
        configs["hf_lm"], configs["lm_hidden_dim"], configs["chunk_size"], configs["chunk_num"]
    )
    init_with_ckpt(model, configs["ckpt_dir"], "torch")
    
    train_dataset: TextOnlyDataset = TextOnlyDataset(
        train_data_path, data_dict_path, tokenizer, 
        text_col=configs["text_col"], label_col=configs["label_col"],
        chunk_size=configs["chunk_size"], chunk_num=configs["chunk_num"], 
        data_format="jsonl",
        label_splitter=configs["label_splitter"]
    )
    dev_dataset: TextOnlyDataset = TextOnlyDataset(
        dev_data_path, data_dict_path, tokenizer,
        text_col=configs["text_col"], label_col=configs["label_col"],
        chunk_size=configs["chunk_size"], chunk_num=configs["chunk_num"],
        data_format="jsonl",
        label_splitter=configs["label_splitter"]
    )
    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=configs["single_worker_batch_size"], shuffle=True
    )
    dev_dataloader: DataLoader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
    
    optimizer: AdamW = AdamW(model.parameters(), lr=configs["lr"])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.6)
   
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
            if batch_id % configs["log_period"]  == 0:
                eval_metrics: Dict[str, float] = evaluation(
                    model, 
                    dev_dataloader, 
                    device, 
                    configs["single_worker_eval_size"], 
                    label_confidence_threshold=configs["eval"]["label_confidence_threshold"]
                )
                eval_metrics["train_loss"] = round(float(loss.detach().cpu()), 6)
                eval_metrics["epoch"] = epoch_id
                eval_metrics["batch"] = batch_id
                eval_metrics["step"] = global_step_id
                eval_metrics["lr"] = format(get_lr(optimizer), "f")
                if configs["training_engine"] == "torch":
                    print(eval_metrics)
                elif configs["training_engine"] == "ray":
                    ray.train.report(metrics=eval_metrics)

            if global_step_id % configs["dump_period"] == 0:
                ckpt_dump(model, global_step_id, batch_id, epoch_id, configs)
            global_step_id += 1
        scheduler.step()

    eval_results: Dict[str, float] = evaluation(
        model, 
        dev_dataloader, 
        device, 
        configs["single_worker_eval_size"], 
        label_confidence_threshold=configs["eval"]["label_confidence_threshold"],
        verbose=True
    )
    print({k: v for k, v in eval_results.items() if k != "verbose"})
    final_ckpt_dir: str = os.path.join(configs["ckpt_dir"], "final")
    ckpt_dump(model, global_step_id, batch_id, epoch_id, configs, final_ckpt_dir)
    os.system("cp %s %s" % (data_dict_path, final_ckpt_dir))
    open(os.path.join(final_ckpt_dir, "eval_results.json"), "w").write(
        json.dumps(eval_results)
    )


if __name__ == "__main__":
    train_conf: Dict = json.loads(open(sys.argv[1], "r").read())
    train_conf["data_dir"] = os.path.abspath(train_conf["data_dir"])
    train_conf["ckpt_dir"] = os.path.abspath(train_conf["ckpt_dir"])
    if os.path.exists(train_conf["hf_lm"]):
        train_conf["hf_lm"] = os.path.abspath(train_conf["hf_lm"])
    print("Training config:\n{}".format(train_conf))
    
    os.environ["HF_TOKEN"] = train_conf["hf_key"]
    os.system("mkdir -p %s" % train_conf["ckpt_dir"])

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

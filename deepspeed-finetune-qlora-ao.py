from project.dataset.collate import DataCollatorWithPadding
from project.dataset.prepare import DatasetPreparer
from project.trainer.lightning import VideoLlavaModelPLModule
from project.trainer.peft import find_all_linear_names

from transformers import (
    VideoLlavaProcessor,
    BitsAndBytesConfig,
    VideoLlavaForConditionalGeneration
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch
from dataclasses import dataclass
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import argparse
import logging
import os
import sys
from datetime import datetime
from lightning.pytorch.strategies import DeepSpeedStrategy

@dataclass
class Config:
    lora_r: int = 4
    lora_alpha: int = 8
    batch_size: int = 1
    max_epoch: int = 2
    val_check_interval: float = 0.25
    learning_rate: float = 2e-5
    dataset_dir: str = "datasets/processed"
    num_frames: int = 14
    num_worker: int = 2
    hub_repo: str = "jwnt4/finetune-videollava-qlora"
    accumulate_grad_batches: int = 1
    limit_val_batches: float = 16

args = Config

deepspeed_config = {
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 2e-5,
            "betas": [0.998, 0.999],
            "eps": 1e-5,
            "weight_decay": 1e-9,
        },
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "last_batch_iteration": -1,
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 100,
        },
    },
    "zero_optimization": {
        "stage": 2,  # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
        "offload_optimizer": {"device": "cpu"},  # Enable Offloading optimizer state/calculation to the host CPU
        "contiguous_gradients": True,  # Reduce gradient fragmentation.
        "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
        "allgather_bucket_size": 2e8,  # Number of elements to all gather at once.
        "reduce_bucket_size": 2e8,  # Number of elements we reduce/allreduce at once.
    }
}

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    
    if not os.path.isdir("logs"):
        os.makedirs("logs")
    
    log_file = f"logs/{str(datetime.now()).replace(' ', '_')}.log"
    file_handler = logging.FileHandler(log_file)
    
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    stream_handler.setFormatter(log_formatter)
    
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", use_fast=False)
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"
    processor.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    base_dir = args.dataset_dir.split("/")[0]
    processed_dir = args.dataset_dir.split("/")[1]
    dp = DatasetPreparer(
        base_dir=base_dir, processed_dir=processed_dir, num_frames=args.num_frames, num_worker=2, processor=processor
    )

    dataset = None
    try:
        dataset = load_from_disk(f"{args.dataset_dir}/action_ordering_v2/robust/{args.num_frames}_frames")
    except:
        dataset = None
    if dataset is None:
        dataset = dp.prepare_dataset('action_ordering_v2', use_robust=True)
        
    train_dataloader = DataLoader(dataset['train'], collate_fn=DataCollatorWithPadding(processor), batch_size=args.batch_size, shuffle=False, num_workers=2)
    eval_dataloader = DataLoader(dataset['validation'], collate_fn=DataCollatorWithPadding(processor), batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Define quantized model in 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        "LanguageBind/Video-LLaVA-7B-hf",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.generation_config.max_new_tokens = 40

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian"
    ) 
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    module = VideoLlavaModelPLModule(
        config={
            "lr": args.learning_rate
        },
        processor=processor,
        model=model
    )
    
    early_stopping = EarlyStopping(monitor="val_accuracy", verbose=False, mode="min")
    model_checkpoint = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath='output/',
        filename='videollava-7b-ao-{epoch:02d}-{val_accuracy:.2f}'+f"lora_r{args.lora_r}-lora_alpha{args.lora_alpha}"
    )
    callbacks = [
        early_stopping, model_checkpoint
    ]

    limit_val_batches = (args.limit_val_batches // args.batch_size) * args.batch_size
    train_conf = {
        "max_epochs": args.max_epoch,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "limit_val_batches": int(limit_val_batches),
        "val_check_interval": args.val_check_interval,
        "precision": "16-mixed",
        "gradient_clip_val": 1.0,
        "num_sanity_val_steps": int(args.batch_size * 8)
    }
    logger.info(str(train_conf))
    
    trainer = Trainer(
        **train_conf,
        accelerator="auto",
        devices=[0],
        callbacks=callbacks,
        strategy=DeepSpeedStrategy(config=deepspeed_config)
    )

    trainer.fit(module, train_dataloader, eval_dataloader)


from project.dataset.prepare import DatasetPreparer
from project.dataset.collate import DataCollatorWithPadding
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

logger = logging.getLogger(__name__)

@dataclass
class Config:
    lora_r: int = 8
    lora_alpha: int = 8
    batch_size: int = 4
    max_epoch: int = 2
    val_check_interval: float = 0.25
    learning_rate: float = 2e-5
    dataset_dir: str = "datasets/processed"
    num_frames: int = 14
    num_worker: int = 2
    hub_repo: str = None
    accumulate_grad_batches: int = 2
    limit_val_batches: float = 32

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse LoRA training parameters.")

    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (integer).")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha (integer).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (integer).")
    parser.add_argument("--max_epoch", type=int, default=2, help="Maximum number of epochs (integer).")
    parser.add_argument("--val_check_interval", type=float, default=0.25, help="Validation check interval (float).")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate (float).")
    parser.add_argument("--dataset_dir", type=str, default="datasets/processed", help="Path to the dataset directory.")
    parser.add_argument("--num_frames", type=int, default=14, help="Number of frames to process (integer).")
    parser.add_argument("--num_worker", type=int, default=2, help="Number of workers for dataset operations (integer).")
    parser.add_argument("--hub_repo", type=str, required=True, help="Repository URL or path to push the results to.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=2, help="Number of batches to accumulate gradients over.")
    parser.add_argument("--limit_val_batches", type=float, default=32, help="Fraction of validation batches to use (float).")

    args = parser.parse_args()

    return Config(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        val_check_interval=args.val_check_interval,
        learning_rate=args.learning_rate,
        dataset_dir=args.dataset_dir,
        num_frames=args.num_frames,
        num_worker=args.num_worker,
        hub_repo=args.hub_repo,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_val_batches=args.limit_val_batches,
    )


if __name__ == "__main__":
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
    args = parse_arguments()

    logger.info(str(args))

    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", use_fast=False)
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"
    processor.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    base_dir = args.dataset_dir.split("/")[0]
    processed_dir = args.dataset_dir.split("/")[1]
    dp = DatasetPreparer(
        base_dir=base_dir, processed_dir=processed_dir, num_frames=args.num_frames, num_worker=2
    )

    dataset = None
    try:
        dataset = load_from_disk(f"{args.dataset_dir}/action_ordering_v2/robust/{args.num_frames}_frames")
    except:
        dataset = None
    if dataset is None:
        dataset = dp.prepare_dataset('action_ordering_v2', use_robust=True)


    train_dataloader = DataLoader(dataset['train'], collate_fn=DataCollatorWithPadding(processor), batch_size=args.batch_size, shuffle=False, num_workers=2)
    eval_dataloader = DataLoader(dataset['test'], collate_fn=DataCollatorWithPadding(processor), batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Define quantized model in 

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = VideoLlavaForConditionalGeneration.from_pretrained(
        "LanguageBind/Video-LLaVA-7B-hf",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto"
    )

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
        "num_sanity_val_steps": int(args.batch_size * 4)
    }

    trainer = Trainer(
        **train_conf,
        accelerator="auto",
        devices=[0],
        callbacks=callbacks,
    )

    trainer.fit(model, train_dataloader, eval_dataloader)
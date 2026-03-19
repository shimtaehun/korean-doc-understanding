"""Florence-2 LoRA 파인튜닝 학습 루프."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset

from src.data.dataset import CORDDataset, CORD_SPECIAL_TOKENS
from src.model.florence_lora import LoRASettings, load_florence_with_lora
from src.training.evaluate import evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict, processor) -> tuple[DataLoader, DataLoader]:
    """CORD v2 데이터셋 로드 및 DataLoader 생성."""
    hf_dataset = load_dataset(cfg["data"]["hf_dataset"])

    train_ds = CORDDataset(
        hf_dataset=hf_dataset["train"],
        processor=processor,
        split="train",
        max_length=cfg["model"]["max_length"],
        augment=True,
    )
    val_ds = CORDDataset(
        hf_dataset=hf_dataset["validation"],
        processor=processor,
        split="validation",
        max_length=cfg["model"]["max_length"],
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    cfg: dict,
    device: str,
    epoch: int,
) -> float:
    """1 에폭 학습 후 평균 loss 반환."""
    model.train()
    total_loss = 0.0
    accum_steps = cfg["training"]["gradient_accumulation_steps"]
    log_interval = cfg["wandb"]["log_interval"]

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(dtype=torch.float16, enabled=cfg["training"]["fp16"]):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["max_grad_norm"]
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

        if (step + 1) % log_interval == 0:
            avg = total_loss / (step + 1)
            lr = scheduler.get_last_lr()[0]
            print(f"  [epoch {epoch} step {step+1}/{len(loader)}] loss={avg:.4f} lr={lr:.2e}")
            wandb.log({"train/loss": avg, "train/lr": lr, "epoch": epoch})

    return total_loss / len(loader)


def save_checkpoint(model, processor, cfg: dict, epoch: int, metric: float, is_best: bool) -> None:
    """체크포인트 저장."""
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"]) / f"epoch_{epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(ckpt_dir))
    processor.save_pretrained(str(ckpt_dir))

    if is_best:
        best_dir = Path(cfg["output"]["checkpoint_dir"]) / "best_model"
        model.save_pretrained(str(best_dir))
        processor.save_pretrained(str(best_dir))
        print(f"  ✓ Best model 저장 (epoch={epoch}, field_f1={metric:.4f})")


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # WandB 초기화
    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        config=cfg,
        name=f"lora-r{cfg['lora']['r']}-lr{cfg['training']['learning_rate']}",
    )

    # 모델 + 프로세서 로드
    lora_settings = LoRASettings(
        r=cfg["lora"]["r"],
        alpha=cfg["lora"]["alpha"],
        dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
    )
    model, processor = load_florence_with_lora(
        model_id=cfg["model"]["name"],
        lora_settings=lora_settings,
        device=device,
        special_tokens=CORD_SPECIAL_TOKENS,
    )

    # DataLoader
    train_loader, val_loader = build_dataloaders(cfg, processor)

    # Optimizer + Scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=cfg["training"]["fp16"])

    # 학습 루프
    best_metric = 0.0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\n=== Epoch {epoch}/{cfg['training']['epochs']} ===")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, device, epoch
        )

        # 검증
        val_metrics = evaluate(model, processor, val_loader, device)
        field_f1 = val_metrics["field_f1"]
        val_cer = val_metrics["cer"]

        print(f"  train_loss={train_loss:.4f} | val_field_f1={field_f1:.4f} | val_cer={val_cer:.4f}")
        wandb.log({
            "val/field_f1": field_f1,
            "val/cer": val_cer,
            "train/epoch_loss": train_loss,
            "epoch": epoch,
        })

        # 체크포인트 저장
        is_best = field_f1 > best_metric
        if is_best:
            best_metric = field_f1

        if not cfg["output"]["save_best_only"] or is_best:
            save_checkpoint(model, processor, cfg, epoch, field_f1, is_best)

    print(f"\n학습 완료. Best Field F1: {best_metric:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    train(args.config)

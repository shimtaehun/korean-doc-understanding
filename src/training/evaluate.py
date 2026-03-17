"""Florence-2 모델 평가 함수."""

import re
from typing import Optional

import torch
from jiwer import cer
from torch.amp import autocast
from torch.utils.data import DataLoader


def extract_fields(text: str) -> dict:
    """XML-like 시퀀스에서 필드 값 추출."""
    fields = {}
    for match in re.finditer(r"<s_(\w+)>(.*?)</s_\1>", text, re.DOTALL):
        key, value = match.group(1), match.group(2).strip()
        if key in fields:
            if isinstance(fields[key], list):
                fields[key].append(value)
            else:
                fields[key] = [fields[key], value]
        else:
            fields[key] = value
    return fields


def compute_field_f1(pred_fields: dict, gt_fields: dict) -> float:
    """필드 단위 F1 계산."""
    all_keys = set(pred_fields.keys()) | set(gt_fields.keys())
    if not all_keys:
        return 0.0

    tp = sum(
        1 for k in all_keys
        if pred_fields.get(k) == gt_fields.get(k) and k in pred_fields and k in gt_fields
    )
    precision = tp / len(pred_fields) if pred_fields else 0.0
    recall = tp / len(gt_fields) if gt_fields else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(
    model,
    processor,
    loader: DataLoader,
    device: str,
    max_new_tokens: int = 512,
) -> dict:
    """검증셋 전체에 대해 Field F1 / CER 계산.

    Args:
        model: 평가할 모델
        processor: Florence-2 processor
        loader: 검증 DataLoader
        device: 디바이스
        max_new_tokens: 생성 최대 토큰 수

    Returns:
        {"field_f1": float, "cer": float}
    """
    model.eval()
    f1_scores, cer_scores = [], []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)  # (B, L), -100 마스킹

            with autocast("cuda", dtype=torch.float16):
                output_ids = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=3,
                )

            # 입력 토큰 제거 후 디코딩
            generated = output_ids[:, input_ids.shape[1]:]
            pred_texts = processor.batch_decode(generated, skip_special_tokens=False)

            # labels에서 -100 제거 후 디코딩 (정답)
            gt_ids = labels.clone()
            gt_ids[gt_ids == -100] = processor.tokenizer.pad_token_id
            gt_texts = processor.batch_decode(gt_ids, skip_special_tokens=False)

            for pred, gt in zip(pred_texts, gt_texts):
                pred_fields = extract_fields(pred)
                gt_fields = extract_fields(gt)
                f1_scores.append(compute_field_f1(pred_fields, gt_fields))
                cer_scores.append(cer(gt, pred) if gt.strip() else 1.0)

    return {
        "field_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "cer": sum(cer_scores) / len(cer_scores) if cer_scores else 1.0,
    }

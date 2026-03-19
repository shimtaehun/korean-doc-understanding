"""Florence-2 모델 평가 함수."""

import re
from typing import Optional

import torch
from jiwer import cer
from torch.amp import autocast
from torch.utils.data import DataLoader


def normalize_xml_tags(text: str) -> str:
    """태그 prefix 오타 정규화: </r_price> → </s_price>, <s-price> → <s_price>"""
    return re.sub(r"<(/?)[a-zA-Z]{0,3}[-,._]", lambda m: f"<{m.group(1)}s_", text)


def extract_fields(text: str) -> dict:
    """XML-like 시퀀스에서 리프 필드 값 추출.

    열기 태그 불일치(예: <s_menu>text</s_nm>)도 닫는 태그 기준으로 추출.
    예: REAL GANACHE</s_nm> → {"nm": "REAL GANACHE"}
    """
    fields = {}
    for match in re.finditer(r"([^<>]+)</s_(\w+)>", text):
        value, key = match.group(1).strip(), match.group(2)
        if not value:
            continue
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

    # Florence-2 decoder forced start: [EOS, BOS, <s_menu>_tokens]
    # 훈련 패턴(EOS→BOS→XML)과 맞춰 XML 생성을 유도
    _menu_ids = processor.tokenizer.encode("<s_menu>", add_special_tokens=False)
    _decoder_prefix = [processor.tokenizer.eos_token_id, processor.tokenizer.bos_token_id] + _menu_ids
    decoder_input_ids = torch.tensor([_decoder_prefix], device=device)

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)  # (B, L), -100 마스킹
            batch_size = pixel_values.shape[0]

            with autocast("cuda", dtype=torch.float16):
                output_ids = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=batch["attention_mask"].to(device),
                    decoder_input_ids=decoder_input_ids.expand(batch_size, -1),
                    max_new_tokens=max_new_tokens,
                    num_beams=3,
                    early_stopping=False,
                )

            # Florence-2: 전체 output 디코딩 (slicing 없이) → XML 태그는 regex로 추출
            pred_texts = processor.batch_decode(output_ids, skip_special_tokens=True)

            # labels에서 -100 제거 후 디코딩 (정답)
            gt_ids = labels.clone()
            gt_ids[gt_ids == -100] = processor.tokenizer.pad_token_id
            gt_texts = processor.batch_decode(gt_ids, skip_special_tokens=True)

            for pred, gt in zip(pred_texts, gt_texts):
                pred_fields = extract_fields(normalize_xml_tags(pred))
                gt_fields = extract_fields(gt)
                f1_scores.append(compute_field_f1(pred_fields, gt_fields))
                cer_scores.append(cer(gt, pred) if gt.strip() else 1.0)

    return {
        "field_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "cer": sum(cer_scores) / len(cer_scores) if cer_scores else 1.0,
    }

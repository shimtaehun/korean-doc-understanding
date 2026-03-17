"""CORD v2 데이터셋 PyTorch Dataset 클래스."""

import json
import re
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


def cord_to_target_sequence(ground_truth: dict) -> str:
    """CORD JSON ground_truth를 Florence-2 학습 타겟 시퀀스로 변환.

    Args:
        ground_truth: CORD gt_parse 딕셔너리

    Returns:
        XML-like 타겟 시퀀스 문자열
    """
    gt_parse = ground_truth.get("gt_parse", {})
    parts = []

    # 상단 정보 (가게명, 날짜 등)
    menu_items = gt_parse.get("menu", [])
    # menu가 dict로 오는 샘플 대응 (CORD v2 일부 샘플)
    if isinstance(menu_items, dict):
        menu_items = [menu_items]
    sub_total = gt_parse.get("sub_total", {})
    total = gt_parse.get("total", {})

    # sub_total / total이 list로 오는 샘플 대응
    if isinstance(sub_total, list):
        sub_total = sub_total[0] if sub_total and isinstance(sub_total[0], dict) else {}
    if isinstance(total, list):
        total = total[0] if total and isinstance(total[0], dict) else {}

    if menu_items:
        parts.append("<s_menu>")
        for item in menu_items:
            if not isinstance(item, dict):
                continue
            parts.append("<s_menuitem>")
            if "nm" in item:
                parts.append(f"<s_nm>{item['nm']}</s_nm>")
            if "price" in item:
                parts.append(f"<s_price>{item['price']}</s_price>")
            if "cnt" in item:
                parts.append(f"<s_cnt>{item['cnt']}</s_cnt>")
            parts.append("</s_menuitem>")
        parts.append("</s_menu>")

    if isinstance(sub_total, dict) and sub_total:
        parts.append("<s_sub_total>")
        for key, value in sub_total.items():
            parts.append(f"<s_{key}>{value}</s_{key}>")
        parts.append("</s_sub_total>")

    if isinstance(total, dict) and total:
        parts.append("<s_total>")
        for key, value in total.items():
            parts.append(f"<s_{key}>{value}</s_{key}>")
        parts.append("</s_total>")

    return "".join(parts)


def parse_model_output(text: str) -> dict:
    """모델 출력 XML-like 시퀀스를 JSON 딕셔너리로 파싱.

    Args:
        text: 모델 출력 문자열

    Returns:
        파싱된 딕셔너리 (실패 시 {"raw": text} 반환)
    """
    try:
        result = {}

        # 메뉴 아이템 파싱
        menu_match = re.search(r"<s_menu>(.*?)</s_menu>", text, re.DOTALL)
        if menu_match:
            items = []
            for item_match in re.finditer(r"<s_menuitem>(.*?)</s_menuitem>", menu_match.group(1), re.DOTALL):
                item = {}
                for field in ["nm", "price", "cnt"]:
                    field_match = re.search(rf"<s_{field}>(.*?)</s_{field}>", item_match.group(1))
                    if field_match:
                        item[field] = field_match.group(1).strip()
                if item:
                    items.append(item)
            result["menu"] = items

        # sub_total 파싱
        sub_total_match = re.search(r"<s_sub_total>(.*?)</s_sub_total>", text, re.DOTALL)
        if sub_total_match:
            sub_total = {}
            for field_match in re.finditer(r"<s_(\w+)>(.*?)</s_\1>", sub_total_match.group(1)):
                sub_total[field_match.group(1)] = field_match.group(2).strip()
            result["sub_total"] = sub_total

        # total 파싱
        total_match = re.search(r"<s_total>(.*?)</s_total>", text, re.DOTALL)
        if total_match:
            total = {}
            for field_match in re.finditer(r"<s_(\w+)>(.*?)</s_\1>", total_match.group(1)):
                total[field_match.group(1)] = field_match.group(2).strip()
            result["total"] = total

        return result if result else {"raw": text}

    except Exception:
        return {"raw": text}


class CORDDataset(Dataset):
    """CORD v2 HuggingFace 데이터셋 래퍼.

    Args:
        hf_dataset: HuggingFace datasets의 split (예: dataset["train"])
        processor: Florence-2 AutoProcessor
        split: "train", "validation", "test"
        max_length: 타겟 시퀀스 최대 토큰 수
        augment: 학습 시 augmentation 적용 여부
    """

    PROMPT = "<DocVQA>"

    def __init__(
        self,
        hf_dataset,
        processor: AutoProcessor,
        split: str = "train",
        max_length: int = 512,
        augment: bool = False,
    ) -> None:
        self.dataset = hf_dataset
        self.processor = processor
        self.split = split
        self.max_length = max_length
        self.augment = augment and (split == "train")

        self._augment_transform = self._build_augment() if self.augment else None

    def _build_augment(self):
        """Albumentations augmentation 파이프라인 구성."""
        try:
            import albumentations as A
            import numpy as np
            from albumentations.pytorch import ToTensorV2

            return A.Compose([
                A.Rotate(limit=5, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(std_range=(0.04, 0.12), p=0.2),
                A.GaussianBlur(blur_limit=(3, 3), p=0.1),
            ])
        except ImportError:
            return None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]

        # 이미지 로드
        image: Image.Image = sample["image"].convert("RGB")

        # Augmentation (train only)
        if self._augment_transform is not None:
            import numpy as np
            img_array = self._augment_transform(image=np.array(image))["image"]
            image = Image.fromarray(img_array)

        # 타겟 시퀀스 생성
        ground_truth = json.loads(sample["ground_truth"])
        target_sequence = cord_to_target_sequence(ground_truth)

        # Florence-2 processor로 인코딩
        # max_length/padding/truncation은 processor에 전달하지 않음:
        # 내부적으로 image token 수를 뺀 길이가 음수가 되어 OverflowError 발생
        encoding = self.processor(
            text=self.PROMPT,
            images=image,
            return_tensors="pt",
        )

        # 타겟 레이블 토크나이징
        labels = self.processor.tokenizer(
            target_sequence,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        # padding 토큰은 loss 계산에서 제외 (-100)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

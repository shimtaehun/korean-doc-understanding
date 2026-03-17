"""CORDDataset 단위 테스트.

실행: pytest tests/test_dataset.py -v

주의: Florence-2 processor 로드가 필요하므로 인터넷 연결 필수.
      모델 다운로드를 피하려면 --mock 옵션 대신 processor만 로드.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.data.dataset import CORDDataset, cord_to_target_sequence, parse_model_output


# ── cord_to_target_sequence 테스트 ──────────────────────────────────────────

class TestCordToTargetSequence:

    def test_basic_structure(self):
        gt = {
            "gt_parse": {
                "menu": [{"nm": "아메리카노", "price": "2,000", "cnt": "1"}],
                "total": {"total_price": "2,000"},
            }
        }
        result = cord_to_target_sequence(gt)
        assert "<s_menu>" in result
        assert "<s_nm>아메리카노</s_nm>" in result
        assert "<s_price>2,000</s_price>" in result
        assert "<s_total>" in result
        assert "<s_total_price>2,000</s_total_price>" in result

    def test_empty_ground_truth(self):
        result = cord_to_target_sequence({})
        assert result == ""

    def test_multiple_menu_items(self):
        gt = {
            "gt_parse": {
                "menu": [
                    {"nm": "삼각김밥", "price": "1,500"},
                    {"nm": "아메리카노", "price": "2,000"},
                ]
            }
        }
        result = cord_to_target_sequence(gt)
        assert result.count("<s_menuitem>") == 2
        assert "삼각김밥" in result
        assert "아메리카노" in result

    def test_missing_optional_fields(self):
        """cnt, price 없이 nm만 있는 아이템도 처리."""
        gt = {
            "gt_parse": {
                "menu": [{"nm": "콜라"}]
            }
        }
        result = cord_to_target_sequence(gt)
        assert "<s_nm>콜라</s_nm>" in result
        assert "<s_price>" not in result

    def test_sub_total_included(self):
        gt = {
            "gt_parse": {
                "sub_total": {"subtotal_price": "3,500", "discount_price": "-500"},
                "total": {"total_price": "3,000"},
            }
        }
        result = cord_to_target_sequence(gt)
        assert "<s_sub_total>" in result
        assert "<s_subtotal_price>3,500</s_subtotal_price>" in result
        assert "<s_discount_price>-500</s_discount_price>" in result


# ── parse_model_output 테스트 ────────────────────────────────────────────────

class TestParseModelOutput:

    def test_roundtrip(self):
        """cord_to_target_sequence → parse_model_output 왕복 변환."""
        gt = {
            "gt_parse": {
                "menu": [{"nm": "아메리카노", "price": "2,000"}],
                "total": {"total_price": "2,000"},
            }
        }
        seq = cord_to_target_sequence(gt)
        parsed = parse_model_output(seq)

        assert "menu" in parsed
        assert parsed["menu"][0]["nm"] == "아메리카노"
        assert "total" in parsed

    def test_malformed_output_returns_raw(self):
        """파싱 불가 입력은 {"raw": ...} 반환."""
        result = parse_model_output("완전히 이상한 텍스트")
        assert "raw" in result

    def test_empty_string(self):
        result = parse_model_output("")
        assert "raw" in result


# ── CORDDataset 테스트 (processor mock) ──────────────────────────────────────

def _make_mock_processor():
    """Florence-2 processor를 흉내 내는 mock 객체."""
    processor = MagicMock()

    # processor(text=..., images=...) 호출 결과
    processor.return_value = {
        "pixel_values": torch.zeros(1, 3, 768, 768),
        "input_ids": torch.zeros(1, 512, dtype=torch.long),
        "attention_mask": torch.ones(1, 512, dtype=torch.long),
    }

    # processor.tokenizer 호출 결과
    tokenizer_output = MagicMock()
    tokenizer_output.input_ids = torch.zeros(1, 512, dtype=torch.long)
    processor.tokenizer.return_value = tokenizer_output
    processor.tokenizer.pad_token_id = 0

    return processor


def _make_mock_hf_dataset(n: int = 5):
    """CORD 데이터셋 샘플을 흉내 내는 리스트."""
    gt = json.dumps({
        "gt_parse": {
            "menu": [{"nm": "아메리카노", "price": "2,000"}],
            "total": {"total_price": "2,000"},
        }
    })
    samples = []
    for _ in range(n):
        samples.append({
            "image": Image.new("RGB", (400, 600), color=(255, 255, 255)),
            "ground_truth": gt,
        })
    return samples


class TestCORDDataset:

    def setup_method(self):
        self.processor = _make_mock_processor()
        self.hf_dataset = _make_mock_hf_dataset(n=5)
        self.dataset = CORDDataset(
            hf_dataset=self.hf_dataset,
            processor=self.processor,
            split="train",
            max_length=512,
            augment=False,
        )

    def test_len(self):
        assert len(self.dataset) == 5

    def test_getitem_keys(self):
        item = self.dataset[0]
        assert set(item.keys()) == {"pixel_values", "input_ids", "attention_mask", "labels"}

    def test_getitem_shapes(self):
        item = self.dataset[0]
        assert item["pixel_values"].shape == (3, 768, 768)
        assert item["input_ids"].shape == (512,)
        assert item["attention_mask"].shape == (512,)
        assert item["labels"].shape == (512,)

    def test_labels_pad_masked(self):
        """패딩 토큰 위치의 labels는 -100이어야 함."""
        item = self.dataset[0]
        # mock에서 labels 전체가 0 (pad_token_id=0) → 전부 -100으로 치환되어야 함
        assert (item["labels"] == -100).all()

    def test_validation_no_augment(self):
        """validation split은 augment가 적용되지 않아야 함."""
        val_ds = CORDDataset(
            hf_dataset=self.hf_dataset,
            processor=self.processor,
            split="validation",
            augment=True,  # split이 validation이면 무시됨
        )
        assert val_ds._augment_transform is None

    def test_dataloader_batch(self):
        """DataLoader로 batch 생성 테스트."""
        from torch.utils.data import DataLoader

        loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        assert batch["pixel_values"].shape == (2, 3, 768, 768)
        assert batch["input_ids"].shape == (2, 512)
        assert batch["labels"].shape == (2, 512)

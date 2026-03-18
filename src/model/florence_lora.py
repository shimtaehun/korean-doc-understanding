"""Florence-2 LoRA 적용 모델 래퍼."""

from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor


@dataclass
class LoRASettings:
    """LoRA 하이퍼파라미터."""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    bias: str = "none"


def load_florence_with_lora(
    model_id: str = "microsoft/Florence-2-base-ft",
    lora_settings: Optional[LoRASettings] = None,
    torch_dtype: torch.dtype = torch.float16,
    device: Optional[str] = None,
) -> tuple:
    """Florence-2에 LoRA를 적용해 반환.

    Args:
        model_id: HuggingFace 모델 ID
        lora_settings: LoRA 설정 (None이면 기본값 사용)
        torch_dtype: 모델 가중치 dtype
        device: 로드할 디바이스 (None이면 자동)

    Returns:
        (model, processor) 튜플
    """
    if lora_settings is None:
        lora_settings = LoRASettings()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,  # Florence-2 필수
        torch_dtype=torch_dtype,
        attn_implementation="eager",  # flash_attn 없이 실행
    ).to(device)

    lora_config = LoraConfig(
        r=lora_settings.r,
        lora_alpha=lora_settings.alpha,
        lora_dropout=lora_settings.dropout,
        target_modules=lora_settings.target_modules,
        bias=lora_settings.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def load_florence_for_inference(
    base_model_id: str,
    lora_checkpoint_path: str,
    torch_dtype: torch.dtype = torch.float16,
    device: Optional[str] = None,
) -> tuple:
    """학습된 LoRA 체크포인트를 베이스 모델에 병합해 추론용으로 반환.

    Args:
        base_model_id: 베이스 모델 HuggingFace ID
        lora_checkpoint_path: 저장된 LoRA 가중치 경로
        torch_dtype: 모델 가중치 dtype
        device: 로드할 디바이스

    Returns:
        (model, processor) 튜플
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )

    model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
    model = model.merge_and_unload()  # LoRA 가중치를 베이스에 병합
    model = model.to(device)
    model.eval()

    return model, processor


def count_parameters(model) -> dict:
    """전체/학습 가능 파라미터 수 반환.

    Returns:
        {"total": int, "trainable": int, "trainable_ratio": float}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_ratio": trainable / total if total > 0 else 0.0,
    }

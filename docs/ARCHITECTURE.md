# 아키텍처 상세 설계

## 모델 아키텍처

### Florence-2 구조
```
Input Image (768x768)
       │
       ▼
┌─────────────────────────┐
│  DaViT Vision Encoder    │  ← 이미지 → 패치 임베딩 → 비전 특징 추출
│  (Dual Attention ViT)    │
└──────────┬──────────────┘
           │ visual embeddings
           ▼
┌─────────────────────────┐
│  Multi-modal Projector   │  ← 비전 특징을 언어 모델 차원으로 매핑
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  BART-style Transformer  │  ← 프롬프트 + 비전 특징 → 텍스트 시퀀스 생성
│  Decoder (Autoregressive)│    (encoder 출력을 cross-attention으로 참조)
└──────────┬──────────────┘
           │
           ▼
"<s_store>GS25</s_store><s_date>2025.03.17</s_date>..."
```

### LoRA 적용 지점
```
Florence-2 모델
├── Vision Encoder (DaViT)
│   └── Attention layers → LoRA 적용 (선택적)
├── Multi-modal Projector
│   └── Linear layers → LoRA 적용
└── Text Decoder (GPT-2)
    └── Attention layers → LoRA 적용 (핵심)
        ├── q_proj ← LoRA
        ├── k_proj ← LoRA
        ├── v_proj ← LoRA
        └── o_proj ← LoRA
```

### LoRA 설정
```python
lora_config = LoraConfig(
    r=8,                          # rank (실험으로 결정)
    lora_alpha=16,                # scaling factor
    target_modules=["q_proj", "v_proj"],  # 적용 레이어
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

## 데이터 파이프라인

### 입출력 포맷
```
Input:
  - 이미지: 영수증/문서 사진 (다양한 크기 → 768x768 리사이즈)
  - 프롬프트: "<OCR>" 또는 "<s_receipt>"

Output (학습 타겟):
  "<s_store>GS25 강남역점</s_store>
   <s_date>2025.03.17</s_date>
   <s_items>
     <s_item><s_name>삼각김밥</s_name><s_price>1,500</s_price></s_item>
     <s_item><s_name>아메리카노</s_name><s_price>2,000</s_price></s_item>
   </s_items>
   <s_total>3,500</s_total>"
```

### 후처리: 모델 출력 → JSON 파싱
```python
# 모델 출력 (XML-like 토큰 시퀀스)를 JSON으로 변환
def parse_output(text: str) -> dict:
    """
    "<s_store>GS25</s_store><s_total>3500</s_total>"
    → {"store": "GS25", "total": "3500"}
    """
    # 정규식으로 태그 파싱
    # 중첩 태그 (items > item) 처리
    # 파싱 실패 시 fallback 처리
```

### Data Augmentation
```python
transforms = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
])
```

## 학습 설정

### 학습 루프 (직접 작성)
```python
# PyTorch Training Loop 핵심 구조
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast():  # Mixed Precision
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        wandb.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
```

### 기본 하이퍼파라미터 (실험으로 조정)
```yaml
training:
  epochs: 10
  batch_size: 4
  gradient_accumulation_steps: 4  # effective batch = 16
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler: cosine
  max_grad_norm: 1.0
  fp16: true

lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: ["q_proj", "v_proj"]

data:
  image_size: 768
  max_length: 512
```

## 평가 메트릭

### 필드 단위 평가
```
- Field-level F1 Score: 각 필드(store, date, total 등)별로 정확히 추출했는지
- Character Error Rate (CER): 추출한 텍스트의 글자 수준 오류율
- Tree Edit Distance: 구조화된 출력의 계층 구조 정확도
```

## 서빙 아키텍처

```
Client (브라우저/앱)
       │
       │ POST /predict (이미지)
       ▼
┌─────────────────────┐
│  FastAPI Server      │
│  ├── 이미지 검증     │
│  ├── 전처리          │
│  ├── 모델 추론       │  ← 양자화된 모델 (INT8 or ONNX)
│  ├── 후처리 (JSON)   │
│  └── 응답 반환       │
└─────────────────────┘
       │
       ▼
{ "store": "...", "date": "...", "items": [...], "total": "..." }
```

## 양자화 전략

```
Full Model (FP32) ~928MB
    │
    ├── FP16 (Mixed Precision) ~464MB  ← 정확도 손실 거의 없음
    │
    ├── INT8 (bitsandbytes)    ~232MB  ← 정확도 손실 1~3%
    │
    └── INT4 (QLoRA style)     ~116MB  ← 정확도 손실 3~5%

목표: 정확도 손실 5% 이내에서 가장 작고 빠른 모델
```

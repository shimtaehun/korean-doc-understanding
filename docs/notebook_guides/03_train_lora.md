# 노트북 03 — Florence-2 LoRA 파인튜닝

> 대응 파일: `notebooks/03_train_lora.ipynb`

---

## 이 노트북의 목적

CORD v2 데이터로 Florence-2를 LoRA 파인튜닝합니다.
02 노트북의 베이스라인(F1=0)이 얼마나 올라가는지 확인하는 것이 핵심 목표예요.

---

## 셀별 설명

### 셀 1~2 — 레포 클론 + 경로 추가
01, 02와 동일.

---

### 셀 3 — 패키지 설치
```python
!pip install -q transformers peft accelerate datasets wandb jiwer scikit-learn albumentations pyyaml
```

**새로 추가된 패키지**
- `peft`: LoRA 구현 라이브러리 (HuggingFace)
- `accelerate`: 분산 학습/혼합 정밀도 지원 라이브러리
- `albumentations`: 이미지 augmentation 라이브러리

---

### 셀 4 — WandB + GPU 확인
```python
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")

print("GPU:", torch.cuda.get_device_name(0))
```

**T4 vs P100**
- T4: 16GB VRAM, Tensor Core 지원 → FP16 학습에 유리
- P100: 16GB VRAM, 구형이지만 안정적
- 둘 다 batch_size=2, accum_steps=8로 설정하면 동작

---

### 셀 5 — Config 로드 및 Kaggle 환경 맞게 수정
```python
with open(".../configs/train_config.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["output"]["checkpoint_dir"] = "/kaggle/working/checkpoints"
cfg["training"]["batch_size"] = 2
cfg["training"]["gradient_accumulation_steps"] = 8
cfg["training"]["epochs"] = 5
```

**왜 config를 코드에서 override하냐?**
- `train_config.yaml`은 로컬/기본 설정
- Kaggle 환경에서는 경로, 배치 크기 등을 다르게 설정해야 함
- `cfg` 딕셔너리를 직접 수정하면 파일을 바꾸지 않아도 됨

**`yaml.safe_load()`란?**
- YAML 파일을 Python 딕셔너리로 변환
- `safe_load`: 악의적인 YAML 코드 실행 방지 (보안)

---

### 셀 6 — 모델 + 데이터 로드
```python
lora_settings = LoRASettings(r=8, alpha=16, dropout=0.05, target_modules=["q_proj", "v_proj"])
model, processor = load_florence_with_lora(model_id=..., lora_settings=..., device=DEVICE)
```

**이 시점에 출력되는 것**
```
trainable params: 7,340,032 || all params: 270,669,824 || trainable%: 2.71
```
→ 전체 파라미터의 2.71%만 학습. 나머지 97.29%는 고정.

---

### 셀 7 — DataLoader 생성
```python
train_ds = CORDDataset(hf_dataset["train"], processor, split="train", augment=True)
val_ds   = CORDDataset(hf_dataset["validation"], processor, split="validation", augment=False)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
```

**DataLoader 파라미터**
| 파라미터 | 설명 |
|---------|------|
| `batch_size=2` | 한 번에 2개 샘플씩 처리 |
| `shuffle=True` | 매 epoch마다 순서를 섞음 (학습 시) |
| `num_workers=2` | 데이터 로딩을 2개 프로세스로 병렬화 |
| `pin_memory=True` | CPU→GPU 전송 속도 향상 |

**train은 augment=True, val은 augment=False인 이유**
- 학습: 다양한 변형으로 일반화 능력 향상
- 검증: 원본 그대로 평가해야 정확한 성능 측정 가능

---

### 셀 8 — Optimizer, Scheduler, Scaler, Callback 설정
```python
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.01)
```

**`filter(lambda p: p.requires_grad, ...)`**
- LoRA 파라미터만 optimizer에 전달
- freeze된 베이스 모델 파라미터는 제외
- `requires_grad=True`인 것만 필터링

```python
total_steps  = len(train_loader) * epochs   # 전체 step 수
warmup_steps = int(total_steps * 0.1)       # 10%를 warmup에 사용
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
```

**학습률 스케줄 계산 예시**
```
train 800개, batch_size=2, epochs=5
→ 1 epoch = 400 steps
→ total_steps = 400 × 5 = 2000
→ warmup_steps = 2000 × 0.1 = 200
```

```python
scaler = GradScaler(enabled=cfg["training"]["fp16"])
```
**GradScaler란?**
- FP16 학습 시 gradient가 너무 작아져서 0이 되는 문제(underflow) 방지
- gradient를 큰 숫자로 스케일링했다가 업데이트 시 되돌림

---

### 셀 9 — 학습 루프 (핵심)
```python
for epoch in range(1, epochs + 1):
    model.train()

    for step, batch in enumerate(train_loader):
        # GPU로 이동
        pixel_values   = batch["pixel_values"].to(DEVICE)
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        # Forward pass
        with autocast(dtype=torch.float16):
            outputs = model(pixel_values=pixel_values,
                           input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels)
            loss = outputs.loss / accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient Accumulation
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
```

**한 step에서 일어나는 일 순서**
```
1. batch를 GPU로 올림
2. Forward: 이미지 + 텍스트 → 모델 → loss 계산
3. loss / accum_steps (gradient accumulation용)
4. Backward: loss.backward() → 각 파라미터의 gradient 계산
5. accum_steps마다:
   a. gradient 클리핑 (max_norm=1.0)
   b. optimizer.step() → 파라미터 업데이트
   c. scheduler.step() → 학습률 업데이트
   d. optimizer.zero_grad() → gradient 초기화
```

**`model.train()` vs `model.eval()`**
- `model.train()`: Dropout, BatchNorm이 학습 모드로 동작
- `model.eval()`: Dropout 비활성화, BatchNorm이 저장된 통계값 사용
- 학습 시작 시 반드시 `.train()`, 평가 시 반드시 `.eval()` 호출

**`autocast`란?**
```python
with autocast(dtype=torch.float16):
    ...
```
- 이 블록 안의 연산을 FP16으로 실행
- 블록 밖은 FP32 유지
- 자동으로 어떤 연산은 FP16, 어떤 건 FP32로 할지 결정

---

### 셀 10 — 체크포인트 확인
```python
for d in sorted(os.listdir(ckpt_dir)):
    size = sum(os.path.getsize(...) for f in ...) / 1024 / 1024
    print(f"{d}: {size:.1f} MB")
```

**저장되는 파일들**
```
/kaggle/working/checkpoints/
├── best_model/
│   ├── adapter_config.json    # LoRA 설정
│   ├── adapter_model.safetensors  # LoRA 가중치 (작음, ~30MB)
│   └── ...
└── epoch_1/, epoch_2/, ...
```

**LoRA 가중치만 저장하는 이유**
- 베이스 모델(Florence-2)은 HuggingFace에 있으므로 저장 불필요
- LoRA A, B 행렬만 저장 → 파일 크기 매우 작음 (vs 전체 모델 900MB)

---

## 이 노트북에서 배울 것

1. LoRA 파인튜닝의 전체 흐름
2. `model.train()` / `model.eval()` 차이
3. `autocast` + `GradScaler`로 FP16 학습
4. Gradient Accumulation 원리
5. Optimizer, Scheduler 설정
6. 체크포인트 저장 전략

---

## 학습 중 확인할 것

**WandB 대시보드에서**
- `train/loss`: 계속 내려가야 정상. 올라가거나 진동하면 lr 낮추기
- `val/field_f1`: epoch마다 올라가야 정상. plateau면 lr decay 확인
- `train/lr`: warmup 후 cosine 곡선으로 감소하는지 확인

**VRAM 부족 시**
```python
cfg["training"]["batch_size"] = 1                   # 배치 크기 줄이기
cfg["training"]["gradient_accumulation_steps"] = 16  # accum 늘려서 보상
```

**학습이 너무 느릴 때**
```python
cfg["training"]["epochs"] = 3   # epoch 줄이기
```

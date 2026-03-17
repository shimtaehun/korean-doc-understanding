# 코드 가이드 — 처음부터 이해하기

> 이 문서는 프로젝트 코드를 공부하기 위한 설명서입니다.
> 파일 순서대로 읽으면 전체 흐름을 이해할 수 있어요.

---

## 목차

1. [전체 흐름 한눈에 보기](#1-전체-흐름-한눈에-보기)
2. [src/data/dataset.py](#2-srcdatadatasetpy)
3. [src/model/florence_lora.py](#3-srcmodelflorence_lorapy)
4. [src/training/train.py](#4-srctrainingtrainpy)
5. [src/training/evaluate.py](#5-srctrainingevaluatepy)
6. [src/training/callbacks.py](#6-srctrainingcallbackspy)
7. [핵심 개념 정리](#7-핵심-개념-정리)

---

## 1. 전체 흐름 한눈에 보기

```
영수증 이미지
     │
     ▼
[dataset.py] 이미지 + 정답 텍스트를 tensor로 변환
     │
     ▼
[florence_lora.py] Florence-2에 LoRA 붙이기
     │
     ▼
[train.py] 학습 루프 실행
     │  ├── 매 step: loss 계산 → 역전파 → 파라미터 업데이트
     │  └── 매 epoch: evaluate.py로 성능 측정, callbacks.py로 저장/로깅
     ▼
체크포인트 저장 + WandB 기록
```

---

## 2. src/data/dataset.py

### 이 파일이 하는 일
모델이 학습할 수 있는 형태로 데이터를 변환합니다.
CORD v2 데이터셋은 `(이미지, JSON)` 쌍으로 구성되어 있는데,
Florence-2는 `(이미지, 프롬프트 텍스트)` → `(정답 텍스트)` 형태로 학습합니다.
그 사이 변환을 담당하는 파일이에요.

---

### `cord_to_target_sequence()` 함수

```python
def cord_to_target_sequence(ground_truth: dict) -> str:
```

**입력**: CORD JSON (`ground_truth`)
```json
{
  "gt_parse": {
    "menu": [{"nm": "아메리카노", "price": "2,000", "cnt": "1"}],
    "total": {"total_price": "2,000"}
  }
}
```

**출력**: Florence-2 학습용 텍스트 시퀀스
```
<s_menu><s_menuitem><s_nm>아메리카노</s_nm><s_price>2,000</s_price><s_cnt>1</s_cnt></s_menuitem></s_menu>
<s_total><s_total_price>2,000</s_total_price></s_total>
```

**왜 이런 포맷?**
Florence-2는 텍스트를 순서대로 생성(autoregressive)합니다.
JSON을 직접 생성하는 것보다 XML-like 태그 시퀀스가 더 안정적으로 학습돼요.
나중에 `parse_model_output()`으로 다시 JSON으로 변환합니다.

**CORD v2 데이터 특이사항 (실제로 발견한 버그)**
```python
# menu가 list가 아닌 dict로 오는 샘플이 있음
if isinstance(menu_items, dict):
    menu_items = [menu_items]

# sub_total, total이 list로 오는 샘플이 있음
if isinstance(sub_total, list):
    sub_total = sub_total[0] if sub_total and isinstance(sub_total[0], dict) else {}
```
→ 실제 데이터를 돌려보기 전엔 알 수 없었던 케이스들. EDA가 중요한 이유!

---

### `parse_model_output()` 함수

```python
def parse_model_output(text: str) -> dict:
```

`cord_to_target_sequence()`의 반대 방향입니다.
모델이 생성한 XML-like 텍스트를 다시 Python dict로 변환해요.

**정규식 핵심**
```python
re.finditer(r"<s_(\w+)>(.*?)</s_\1>", text, re.DOTALL)
```
- `(\w+)` : 태그 이름 캡처 (예: `nm`, `price`)
- `(.*?)` : 태그 사이 값 캡처 (non-greedy, 가장 짧게 매칭)
- `\1` : 첫 번째 캡처그룹 재참조 (여는 태그 = 닫는 태그)
- `re.DOTALL` : `.`이 줄바꿈도 포함

---

### `CORDDataset` 클래스

```python
class CORDDataset(Dataset):
```

PyTorch의 `Dataset`을 상속받는 클래스입니다.

**PyTorch Dataset의 규칙**
```python
def __len__(self):   # 전체 샘플 수 반환
def __getitem__(self, idx):  # idx번째 샘플 반환
```
이 두 메서드만 구현하면 `DataLoader`가 자동으로 배치를 만들어줘요.

**`__getitem__` 핵심 흐름**
```python
def __getitem__(self, idx):
    # 1. 이미지 + 정답 JSON 가져오기
    image = sample["image"].convert("RGB")
    ground_truth = json.loads(sample["ground_truth"])

    # 2. 정답 시퀀스 변환
    target_sequence = cord_to_target_sequence(ground_truth)

    # 3. Florence-2 processor로 인코딩 (이미지 + 프롬프트 텍스트)
    encoding = self.processor(text=PROMPT, images=image, ...)

    # 4. 정답 텍스트 토크나이징
    labels = self.processor.tokenizer(target_sequence, ...)

    # 5. 패딩 위치는 loss 계산 제외 (-100)
    labels[labels == pad_token_id] = -100

    return {"pixel_values": ..., "input_ids": ..., "labels": ...}
```

**왜 labels에서 -100을 쓰냐?**
PyTorch의 CrossEntropyLoss는 `-100`인 위치를 자동으로 무시합니다.
패딩 토큰 위치에서는 loss를 계산하면 안 되니까요.

---

## 3. src/model/florence_lora.py

### 이 파일이 하는 일
Florence-2에 LoRA를 붙이는 코드입니다.

---

### LoRA란?

**전체 파인튜닝의 문제**
- Florence-2는 파라미터가 약 230M개
- 전부 학습시키려면 VRAM이 엄청나게 필요

**LoRA 아이디어**
```
원래 가중치 행렬 W (고정)
       +
작은 행렬 A × B (학습)
= W + AB
```
- W는 고정, A와 B만 학습
- A: (d × r), B: (r × d) — r이 rank
- rank=8이면 학습 파라미터가 전체의 약 0.5%

**직관적으로**
"모델의 핵심 지식(W)은 그대로 두고,
우리 태스크에 맞게 약간의 조정(AB)만 추가한다"

---

### `LoRASettings` 데이터클래스

```python
@dataclass
class LoRASettings:
    r: int = 8           # rank — 높을수록 표현력↑, 파라미터 수↑
    alpha: int = 16      # scaling = alpha/r, 학습 안정성에 영향
    dropout: float = 0.05  # 과적합 방지
    target_modules: list = ["q_proj", "v_proj"]  # LoRA 적용할 레이어
```

**target_modules 선택 기준**
- `q_proj`, `v_proj`: attention의 Query, Value 행렬 — 핵심
- `k_proj`, `o_proj`: Key, Output — 추가하면 성능 오를 수 있지만 파라미터 증가
- `fc1`, `fc2`: FFN 레이어 — 더 추가

2주차 실험 3에서 어떤 조합이 최적인지 테스트할 예정.

---

### `load_florence_with_lora()` 함수

```python
base_model = AutoModelForCausalLM.from_pretrained(...)  # 베이스 모델 로드
lora_config = LoraConfig(r=8, ...)                       # LoRA 설정
model = get_peft_model(base_model, lora_config)          # LoRA 적용
model.print_trainable_parameters()                       # 학습 파라미터 확인
```

`get_peft_model()`이 내부적으로:
1. `target_modules`에 해당하는 레이어를 찾아서
2. 원본 가중치를 freeze (requires_grad=False)
3. LoRA A, B 행렬을 추가 (requires_grad=True)

---

### `load_florence_for_inference()` 함수

학습이 끝난 후 추론에 사용합니다.

```python
model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
model = model.merge_and_unload()  # LoRA 가중치를 베이스에 병합
```

**merge_and_unload()가 하는 일**
`W + AB`를 계산해서 새로운 `W'`을 만들고, A/B 행렬은 제거합니다.
추론 시 별도의 LoRA 연산 없이 단일 행렬로 동작 → 속도 향상.

---

## 4. src/training/train.py

### 이 파일이 하는 일
전체 학습 루프를 실행합니다.

---

### `train_one_epoch()` 함수 핵심 흐름

```python
for step, batch in enumerate(loader):
    # 1. Forward pass (Mixed Precision)
    with autocast(dtype=torch.float16):
        outputs = model(pixel_values=..., input_ids=..., labels=...)
        loss = outputs.loss / accum_steps  # gradient accumulation

    # 2. Backward pass
    scaler.scale(loss).backward()

    # accum_steps마다 실제 업데이트
    if (step + 1) % accum_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
```

**Mixed Precision (FP16) 이란?**
- 기본적으로 PyTorch는 FP32 (32bit) 연산
- FP16 (16bit)으로 하면 메모리 절반, 속도 2배
- 단, FP16은 overflow가 생길 수 있어서 `GradScaler`로 보정

**Gradient Accumulation 이란?**
```
batch_size=2, accum_steps=8
→ 실제로 2개씩 처리하지만, 8번 모아서 한 번에 업데이트
→ 효과적인 batch_size = 2 × 8 = 16
```
VRAM이 적을 때 큰 배치 효과를 내는 방법.

**Gradient Clipping 이란?**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
gradient가 너무 크면 학습이 발산할 수 있어요.
gradient의 norm이 1.0을 넘으면 1.0으로 잘라냅니다.

---

### Optimizer와 Scheduler

```python
optimizer = AdamW(...)  # Adam + Weight Decay
scheduler = get_cosine_schedule_with_warmup(...)
```

**AdamW**
- Adam에 Weight Decay를 올바르게 적용한 버전
- Weight Decay: 가중치가 너무 커지지 않게 패널티

**Cosine Schedule with Warmup**
```
학습률
  ↑
  │    /\
  │   /  \
  │  /    \__________
  │ /
  └─────────────────→ step
  [warmup] [cosine decay]
```
- 처음엔 낮게 시작 (warmup) → 안정적으로 학습 시작
- 이후 코사인 곡선으로 천천히 감소

---

## 5. src/training/evaluate.py

### 이 파일이 하는 일
검증셋에 대해 Field F1과 CER을 계산합니다.

---

### Field F1

```python
def compute_field_f1(pred_fields: dict, gt_fields: dict) -> float:
    tp = sum(1 for k in all_keys
             if pred_fields.get(k) == gt_fields.get(k)
             and k in pred_fields and k in gt_fields)
    precision = tp / len(pred_fields)
    recall = tp / len(gt_fields)
    return 2 * precision * recall / (precision + recall)
```

**예시**
```
정답:  {"nm": "아메리카노", "price": "2,000", "total": "2,000"}
예측:  {"nm": "아메리카노", "price": "2,500", "total": "2,000"}

TP = 2 (nm, total 일치)
Precision = 2/3 = 0.67
Recall    = 2/3 = 0.67
F1        = 0.67
```

---

### CER (Character Error Rate)

```
CER = (삽입 + 삭제 + 치환) / 정답 글자 수
```

**예시**
```
정답: "아메리카노"  (5글자)
예측: "아메리가노"  (치환 1개)
CER = 1/5 = 0.20  (20% 오류)
```

CER이 0에 가까울수록 좋고, 1 이상이면 거의 다 틀린 것.

---

## 6. src/training/callbacks.py

### 이 파일이 하는 일
학습 중 발생하는 이벤트(step 종료, epoch 종료)에 반응하는 코드입니다.

**콜백 패턴이란?**
학습 루프 안에 WandB 로깅, 체크포인트 저장 코드를 직접 넣으면 지저분해져요.
콜백 객체를 만들어서 "이 이벤트가 발생하면 이 함수를 호출해"라고 등록하는 패턴.

```python
# train.py에서
wandb_cb.on_step_end(loss, lr, epoch)    # step마다
wandb_cb.on_epoch_end(epoch, loss, metrics)  # epoch마다

ckpt_cb.on_epoch_end(model, processor, epoch, metrics)  # best이면 저장
```

---

### `CheckpointCallback`의 best 판단 로직

```python
def is_better(self, current: float) -> bool:
    if self.higher_is_better:
        return current > self._best_metric  # F1: 높을수록 좋음
    return current < self._best_metric      # CER: 낮을수록 좋음
```

`metric_for_best="field_f1"`, `higher_is_better=True`로 설정되어 있어서
Field F1이 이전 최고값보다 높을 때만 `best_model`을 저장합니다.

---

## 7. 핵심 개념 정리

### Florence-2 구조 요약
```
이미지 → DaViT Vision Encoder → 비전 임베딩
프롬프트 텍스트 → 토크나이저 → 텍스트 임베딩
                                    ↓
                         BART-style Decoder
                         (cross-attention으로 비전 임베딩 참조)
                                    ↓
                         "<s_menu><s_menuitem>..."
```

### LoRA가 적용되는 위치
```
Decoder의 Attention 레이어
├── q_proj (Query)  ← LoRA A, B 추가
├── k_proj (Key)
├── v_proj (Value)  ← LoRA A, B 추가
└── o_proj (Output)
```

### 학습 시 메모리 절약 기법 요약
| 기법 | 효과 |
|------|------|
| LoRA | 학습 파라미터 99% 감소 |
| FP16 Mixed Precision | VRAM 50% 감소 |
| Gradient Accumulation | 작은 배치로 큰 배치 효과 |
| Gradient Checkpointing | 중간 활성화 값 재계산으로 메모리 절약 (선택적) |

### 자주 쓰는 용어
| 용어 | 의미 |
|------|------|
| epoch | 전체 학습 데이터를 한 번 다 본 것 |
| step | 배치 하나를 처리한 것 |
| loss | 예측이 정답과 얼마나 다른지 (낮을수록 좋음) |
| overfitting | 학습 데이터엔 잘 맞지만 새 데이터엔 못 맞추는 현상 |
| warmup | 학습률을 처음에 낮게 시작해서 점점 올리는 것 |
| gradient | loss를 파라미터로 미분한 값 (어느 방향으로 업데이트할지) |
| rank (r) | LoRA의 중간 차원 크기. 클수록 표현력↑, 파라미터↑ |

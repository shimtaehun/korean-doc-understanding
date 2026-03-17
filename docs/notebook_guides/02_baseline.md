# 노트북 02 — Florence-2 Zero-shot 베이스라인

> 대응 파일: `notebooks/02_baseline.ipynb`

---

## 이 노트북의 목적

파인튜닝 **없이** Florence-2가 CORD 영수증을 얼마나 잘 읽는지 측정합니다.
이 수치가 "출발점(baseline)"이 되고, 이후 LoRA 파인튜닝 후 얼마나 올랐는지 비교합니다.

**예상 결과**: Field F1 ≈ 0, CER ≈ 1.0
→ 파인튜닝 없으면 우리가 원하는 포맷으로 출력하지 못함

---

## 셀별 설명

### 셀 1~2 — 레포 클론 + 경로 추가
01 노트북과 동일. 매번 최신 코드를 받아옴.

---

### 셀 3 — 패키지 설치
```python
!pip install -q "transformers==4.41.0" "tokenizers==0.19.1" datasets wandb jiwer scikit-learn
```
- `transformers==4.41.0`: Florence-2와 호환되는 버전으로 고정
  - 최신 버전은 `additional_special_tokens` 관련 AttributeError 발생
- `jiwer`: CER 계산 라이브러리

---

### 셀 4 — flash_attn mock
```python
import sys, types, importlib.machinery
import transformers.utils.import_utils

for mod_name in ['flash_attn', 'flash_attn.flash_attn_interface', 'flash_attn.bert_padding']:
    mock = types.ModuleType(mod_name)
    mock.__spec__ = importlib.machinery.ModuleSpec(mod_name, loader=None)
    sys.modules[mod_name] = mock

transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
```

**왜 이게 필요하냐?**
- Florence-2 모델 코드(`modeling_florence2.py`)가 `flash_attn`을 import하려고 함
- Kaggle T4/P100에는 `flash_attn`이 설치되어 있지 않음
- 설치하려면 컴파일이 필요해서 30분 이상 걸림

**해결 방법**
1. `sys.modules`에 가짜(mock) `flash_attn` 모듈 등록
   → `transformers`의 import 검사를 통과시킴
2. `is_flash_attn_2_available = lambda: False` 패치
   → 실제 모델 코드에서 flash_attn 사용 안 하도록 강제
3. 모델 로드 시 `attn_implementation="eager"` 옵션
   → 표준 attention 구현 사용

**`types.ModuleType`이란?**
- 빈 Python 모듈 객체를 코드로 생성하는 방법
- 실제 기능은 없지만 "이 이름의 모듈이 존재한다"는 것만 알려줌

---

### 셀 5 — WandB 초기화
```python
from kaggle_secrets import UserSecretsClient
api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
wandb.login(key=api_key, relogin=True)

run = wandb.init(
    project="korean-doc-understanding",
    name="baseline-zero-shot",
    config={...},
)
```

**WandB run이란?**
- 하나의 실험 = 하나의 run
- `wandb.init()`으로 시작, `wandb.finish()`로 종료
- `config`에 실험 설정을 기록해두면 나중에 비교하기 편함

**Kaggle Secrets를 쓰는 이유**
- API 키를 코드에 직접 넣으면 GitHub에 올렸을 때 노출됨
- Secrets에 저장하면 코드엔 키가 없고 실행 시에만 주입됨

---

### 셀 6 — 모델 로드
```python
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="eager",
).to(DEVICE)
```

**`trust_remote_code=True`가 필요한 이유**
- Florence-2는 HuggingFace 표준 모델이 아니라 커스텀 코드 포함
- 이 옵션 없으면 커스텀 코드 실행을 거부함

**`torch_dtype=torch.float16`**
- 모델 가중치를 FP16으로 로드 → VRAM 절반
- T4 16GB에서 FP32로 로드하면 메모리 부족 가능성

**`AutoProcessor`란?**
- 이미지 전처리 + 텍스트 토크나이저를 하나로 묶은 객체
- `processor(text=..., images=...)` 한 번 호출로 둘 다 처리

---

### 셀 7 — 추론 함수
```python
def run_inference(image, prompt="<DocVQA>", max_new_tokens=512):
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, torch.float16)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=3,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]  # 입력 토큰 제거
    return processor.batch_decode(generated, skip_special_tokens=False)[0]
```

**`torch.no_grad()`란?**
- 추론 시엔 gradient 계산이 불필요
- `no_grad()` 블록 안에선 gradient를 계산하지 않아서 메모리 절약 + 속도 향상

**`model.generate()`란?**
- 텍스트를 한 토큰씩 자동으로 생성하는 함수
- `max_new_tokens=512`: 최대 512 토큰까지 생성
- `num_beams=3`: Beam Search — 3개 후보를 동시에 탐색해서 가장 좋은 결과 선택

**`output_ids[:, inputs["input_ids"].shape[1]:]`**
- `generate()`는 입력 토큰 + 생성 토큰을 모두 반환
- 우리가 원하는 건 생성 토큰만 → 입력 길이 이후부터 슬라이싱

**`skip_special_tokens=False`**
- `<s_menu>` 같은 태그가 special token으로 등록되어 있을 수 있음
- False로 설정해야 태그가 출력에 포함됨

---

### 셀 8 — 평가 함수
```python
def compute_field_f1(pred_fields, gt_fields):
    tp = sum(1 for k in all_keys
             if pred_fields.get(k) == gt_fields.get(k)
             and k in pred_fields and k in gt_fields)
    precision = tp / len(pred_fields)
    recall = tp / len(gt_fields)
    return 2 * precision * recall / (precision + recall)
```

**Precision vs Recall**
```
Precision = 내가 맞다고 한 것 중 실제로 맞은 비율
Recall    = 실제 정답 중 내가 맞춘 비율
F1        = Precision과 Recall의 조화평균
```

**예시**
```
정답 필드: {nm, price, total} (3개)
예측 필드: {nm, price, cnt}  (3개)
일치:       {nm, price}       (2개)

Precision = 2/3 = 0.67
Recall    = 2/3 = 0.67
F1        = 0.67
```

---

### 셀 9 — 추론 실행 루프
```python
for i, sample in enumerate(test_samples):
    gt_sequence = cord_to_target_sequence(json.loads(sample["ground_truth"]))
    pred_sequence = run_inference(sample["image"])

    field_f1 = compute_field_f1(...)
    sample_cer = compute_cer(...)

    wandb.log({"sample_field_f1": field_f1, "sample_cer": sample_cer, "step": i})
```

**`enumerate()`란?**
- 리스트를 순회할 때 인덱스(i)와 값(sample)을 동시에 가져오는 함수

**`wandb.log()`**
- 딕셔너리 형태로 메트릭을 기록
- WandB 대시보드에서 실시간으로 확인 가능

---

### 셀 10 — 결과 집계 및 시각화
```python
avg_f1 = sum(r["field_f1"] for r in results) / len(results)
```

**제너레이터 표현식**
- `(expression for item in iterable)` 형태
- 리스트 컴프리헨션과 비슷하지만 메모리 효율적

---

## 이 노트북에서 배울 것

1. `torch.no_grad()`로 추론 시 메모리 절약
2. `model.generate()` — Beam Search로 텍스트 생성
3. Precision, Recall, F1 계산 원리
4. CER(Character Error Rate) 개념
5. WandB에 실험 결과 로깅하는 방법
6. flash_attn mock으로 환경 문제 우회하는 방법

---

## 실행 결과 해석

| 결과 | 의미 |
|------|------|
| Field F1 = 0.000 | 파인튜닝 없이는 우리 포맷으로 출력 못 함 → 정상 |
| CER = 1.000 | 정답과 출력이 완전히 다름 → 정상 |

**이 결과가 나와야 프로젝트가 제대로 진행되는 것입니다.**
LoRA 파인튜닝 후 Field F1이 올라가는 것이 우리 목표예요.

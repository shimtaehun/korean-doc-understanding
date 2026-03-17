# 노트북 04 — 하이퍼파라미터 탐색

> 대응 파일: `notebooks/04_hyperparameter_search.ipynb`

---

## 이 노트북의 목적

03 노트북으로 학습이 가능하다는 걸 확인했으니,
이번엔 어떤 설정이 가장 좋은 성능을 내는지 비교합니다.

**실험 3가지**
1. LoRA rank (4, 8, 16, 32)
2. 학습률 (1e-4, 5e-5, 1e-5)
3. LoRA target modules (attention only vs attention+FFN)

**핵심 전략**: 각 실험을 epoch=3으로 빠르게 돌려서 비교 → Kaggle 30시간 제한 내 처리

---

## 셀별 설명

### 셀 1~6 — 환경 세팅
02, 03 노트북과 동일 (클론, 경로, 패키지, flash_attn mock, WandB).

---

### 셀 7 — base_cfg 설정

```python
base_cfg["training"]["epochs"] = 3  # 빠른 비교를 위해 3 epoch
hf_dataset = load_dataset(...)       # 데이터셋은 한 번만 로드
```

**왜 데이터셋을 한 번만 로드하냐?**
- `load_dataset()`은 처음엔 다운로드가 필요하지만 이후엔 캐시에서 로드
- 실험마다 다시 로드하면 시간 낭비

---

### 셀 8 — `run_experiment()` 함수

모든 실험이 공통으로 쓰는 함수입니다.

```python
def run_experiment(exp_name: str, lora_settings: LoRASettings, cfg: dict) -> dict:
```

**함수 구조**
```
1. 모델 로드 (lora_settings 기반)
2. DataLoader 생성
3. wandb.init(reinit=True)  ← 실험마다 새 run 시작
4. 학습 루프 실행
5. wandb.finish()
6. 메모리 해제 (del model + torch.cuda.empty_cache())
7. {"exp_name": ..., "best_f1": ...} 반환
```

**`reinit=True`가 중요한 이유**
- 기본적으로 WandB는 run이 이미 있으면 오류 발생
- `reinit=True`로 설정하면 이전 run을 종료하고 새 run 시작 가능
- 여러 실험을 루프로 돌릴 때 필수

**`torch.cuda.empty_cache()`**
- 모델을 삭제해도 GPU 메모리가 즉시 해제되지 않을 수 있음
- `del model` 후 `empty_cache()`로 GPU 메모리 명시적 해제
- 다음 실험에서 메모리 부족 방지

**`copy.deepcopy(base_cfg)`**
- 실험마다 cfg를 수정하는데, 원본 base_cfg가 바뀌면 안 됨
- `deepcopy`로 완전히 독립된 복사본 생성
- 얕은 복사(`copy.copy`)는 중첩 딕셔너리를 공유하므로 부적합

---

### 셀 9 — 실험 1: LoRA rank 비교

```python
for rank in [4, 8, 16, 32]:
    lora = LoRASettings(
        r=rank,
        alpha=rank * 2,  # alpha = rank * 2 관례
        ...
    )
```

**rank란?**
```
원래 가중치 변환: W (d×d)
LoRA 분해:       A (d×r) × B (r×d)

rank=4:  A(d×4)  × B(4×d)   → 파라미터 적음, 표현력 낮음
rank=32: A(d×32) × B(32×d)  → 파라미터 많음, 표현력 높음
```

**alpha = rank * 2 관례**
- `scaling = alpha / rank` → alpha=rank*2 이면 scaling=2.0으로 고정
- rank를 바꿔도 실제 업데이트 크기가 일정하게 유지됨
- 공정한 비교를 위해 rank마다 alpha도 같이 조정

**예상 결과**
- rank=4: 과소적합 가능성 (표현력 부족)
- rank=32: 과적합 가능성 + 학습 시간 증가
- rank=8 or 16: 최적 예상

---

### 셀 10 — 실험 2: 학습률 비교

```python
for lr in [1e-4, 5e-5, 1e-5]:
    cfg["training"]["learning_rate"] = lr
```

**학습률이 미치는 영향**
| 학습률 | 예상 동작 |
|--------|---------|
| 1e-4 (높음) | 빠르게 수렴하지만 발산 위험, 최솟값 놓칠 수 있음 |
| 5e-5 (중간) | Florence-2 파인튜닝 레퍼런스 권장값 |
| 1e-5 (낮음) | 안정적이지만 3 epoch에선 충분히 학습 못 할 수 있음 |

**실험 1의 결과를 먼저 적용**
```python
BEST_RANK = 8  # ← 실험 1 결과 보고 수동으로 수정
```
실험 1이 끝난 후 최적 rank를 여기에 입력해야 함.

---

### 셀 11 — 실험 3: target modules 비교

```python
module_experiments = [
    ("attn-qv",   ["q_proj", "v_proj"]),           # 기본
    ("attn-full", ["q_proj", "k_proj", "v_proj", "o_proj"]),  # attention 전체
    ("attn-ffn",  ["q_proj", "v_proj", "fc1", "fc2"]),        # attention + FFN
]
```

**각 모듈의 역할**
| 모듈 | 역할 |
|------|------|
| q_proj | Query — "무엇을 찾을지" |
| k_proj | Key — "어디서 찾을지" |
| v_proj | Value — "무엇을 가져올지" |
| o_proj | Output — attention 결과 통합 |
| fc1, fc2 | FFN — 비선형 변환 |

**일반적인 경향**
- `q_proj + v_proj`만으로도 대부분의 경우 충분
- `k_proj, o_proj` 추가 시 성능 소폭 향상, 파라미터 2배
- `fc1, fc2` 추가 시 성능 향상되지만 학습 느려짐

---

### 셀 12 — 결과 요약

```python
df = pd.DataFrame(all_results).sort_values("best_f1", ascending=False)
```

**`sort_values(ascending=False)`**
- F1 높은 순서로 정렬
- 어떤 실험이 가장 좋았는지 한눈에 파악

마지막 출력 내용을 `docs/EXPERIMENTS.md`에 복사해서 기록.

---

## 이 노트북에서 배울 것

1. `copy.deepcopy()` — 딕셔너리 완전 복사
2. `torch.cuda.empty_cache()` — GPU 메모리 관리
3. `wandb.init(reinit=True)` — 루프 내 다중 실험
4. LoRA rank, alpha, target_modules의 의미와 trade-off
5. 학습률이 학습에 미치는 영향
6. `pd.DataFrame`으로 실험 결과 정리

---

## GPU 예산 (Kaggle 30시간 기준)

| 실험 | 횟수 | 예상 시간 |
|------|------|---------|
| rank 실험 (3 epoch) | 4회 | ~4시간 |
| lr 실험 (3 epoch) | 3회 | ~3시간 |
| target modules (3 epoch) | 3회 | ~3시간 |
| **합계** | **10회** | **~10시간** |

→ 주당 30시간 중 10시간 사용, 이후 최적 설정으로 full 학습에 여유 있음

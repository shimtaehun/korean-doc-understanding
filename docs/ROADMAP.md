# 프로젝트 로드맵 & 체크리스트

## 학습 환경: Kaggle / Google Colab

> **로컬에서는 코드 작성/테스트만, 실제 학습은 클라우드에서 실행**

### 클라우드 환경 선택 가이드
| 항목 | Kaggle | Google Colab Pro |
|------|--------|-----------------|
| GPU | T4 16GB / P100 16GB | T4 / A100 (Pro+) |
| 주당 GPU 제한 | ~30시간 | 제한 없음 (크레딧 소진 시까지) |
| 데이터 저장 | Kaggle Dataset / Output | Google Drive |
| 세션 유지 | 최대 12시간 | 최대 12시간 (Pro) |
| 비용 | 무료 | 무료 (Pro: 월 $10) |
| 추천 용도 | 실험 단발성 실행 | 긴 학습 세션 |

### 클라우드 환경 공통 주의사항
- **체크포인트는 반드시 외부 저장소에 즉시 백업** (Drive or Kaggle Output)
- Kaggle: 주당 30시간 GPU 제한 → 실험당 예상 시간 미리 계산
- WandB API Key: 노트북 시작 시 `os.environ["WANDB_API_KEY"]` 설정 또는 Kaggle/Colab Secrets 사용
- 세션 종료 시 `/kaggle/working/` 또는 `/content/` 내 파일 소실 주의

### 체크포인트 저장 전략 (필수)
```python
# Kaggle: output 폴더에 저장 → 자동 보존
save_path = "/kaggle/working/checkpoints/"

# Colab: Google Drive에 저장
from google.colab import drive
drive.mount('/content/drive')
save_path = "/content/drive/MyDrive/korean-doc/checkpoints/"
```

### 노트북 구조
`.sh` 스크립트 대신 Jupyter Notebook으로 실행:
```
notebooks/
├── 01_data_exploration.ipynb    # EDA
├── 02_baseline.ipynb            # Zero-shot 베이스라인
├── 03_train_lora.ipynb          # LoRA 파인튜닝
├── 04_hyperparameter_search.ipynb
├── 05_quantization.ipynb
└── 06_export_serve.ipynb
```

---

## 1주차: 환경 세팅 + 데이터 준비 + 베이스라인

### Day 1: 환경 세팅 + AI Hub 신청 (즉시 시작!)
- [ ] **[최우선] AI Hub 한국어 OCR 데이터 다운로드 신청** (승인 1~3주 소요, 먼저 해야 함)
  - https://www.aihub.or.kr → "야외 실제 촬영 한국어 이미지" 또는 "영수증 OCR" 검색
- [ ] 로컬 개발 환경 세팅 (코드 작성용)
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- [ ] WandB 계정 생성 + API Key 발급 (https://wandb.ai)
- [ ] Kaggle 계정 + GPU 활성화 (Phone 인증 필요)
- [ ] GitHub 레포 생성 + 초기 코드 push

### Day 2: CORD 데이터셋 로드 + EDA
- [ ] CORD 데이터셋 HuggingFace에서 로드 (`naver-clova-ix/cord-v2`)
  ```python
  from datasets import load_dataset
  dataset = load_dataset("naver-clova-ix/cord-v2")
  # train: 800개, validation: 100개, test: 100개
  ```
- [ ] EDA: 필드 분포, 이미지 크기, 라벨 통계 분석
- [ ] `notebooks/01_data_exploration.ipynb` 작성
- [ ] 분석 결과 docs/EXPERIMENTS.md에 기록

### Day 3: 데이터 파이프라인
- [ ] PyTorch Dataset 클래스 작성 (`src/data/dataset.py`)
  - CORD JSON → Florence-2 입출력 포맷 변환
  - 이미지 전처리 (768x768 리사이즈, 정규화)
- [ ] Florence-2 토크나이저에 맞는 타겟 시퀀스 포맷 구현
- [ ] DataLoader 테스트 (batch_size=2로 빠른 검증)

### Day 4: 베이스라인 (Kaggle/Colab에서 실행)
- [ ] `notebooks/02_baseline.ipynb` 작성
- [ ] Florence-2-base-ft 모델 로드 테스트
  ```python
  # florence-2는 AutoModelForCausalLM 아님 → 전용 클래스 사용
  from transformers import AutoProcessor, AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained(
      "microsoft/Florence-2-base-ft",
      trust_remote_code=True  # 필수!
  )
  ```
- [ ] Zero-shot 추론 (파인튜닝 없이 CORD 테스트셋 10~20개 샘플)
- [ ] 베이스라인 성능 측정 (Field F1, CER)
- [ ] 결과 WandB + EXPERIMENTS.md에 기록

### Day 5: 학습 코드 준비
- [ ] LoRA 적용 코드 초안 작성 (`src/model/florence_lora.py`)
- [ ] PyTorch Training Loop 초안 작성 (`src/training/train.py`)
- [ ] `notebooks/03_train_lora.ipynb` 작성 (Kaggle/Colab 실행용)
- [ ] 첫 학습 실행 테스트 (epoch=1, 소수 배치로 오류 없는지 확인)

### 1주차 산출물
- [ ] 베이스라인 성능 리포트 (EXPERIMENTS.md)
- [ ] 전처리된 CORD 데이터셋 파이프라인
- [ ] WandB 프로젝트 대시보드 (run 1개 이상)
- [ ] Kaggle/Colab 실행 가능한 학습 노트북

---

## 2주차: LoRA 파인튜닝 + 실험

> **Kaggle 주당 30시간 제한 고려 → 실험 1회 약 1~2시간 예상 → 최대 10~15회 가능**

### Day 1: LoRA 세팅 완성
- [ ] PEFT/LoRA 설정 코드 완성 (`src/model/florence_lora.py`)
- [ ] PyTorch Training Loop 완성 (`src/training/train.py`)
  - optimizer (AdamW)
  - lr scheduler (cosine with warmup)
  - gradient clipping
  - mixed precision (AMP)
- [ ] WandB 로깅 콜백 연동
- [ ] **체크포인트 자동 저장 코드 포함** (Drive / Kaggle output)

### Day 2: 1차 파인튜닝
- [ ] CORD 데이터 전체로 파인튜닝 (LoRA rank=8, lr=5e-5, epoch=5)
- [ ] 학습 곡선 확인 (loss, eval metric)
- [ ] 오버피팅 여부 체크
- [ ] **예상 GPU 사용량 확인** (VRAM 사용률, 학습 시간)

### Day 3: 하이퍼파라미터 실험
- [ ] 실험 1: LoRA rank (4, 8, 16, 32) — 각 rank 당 3 epoch으로 빠르게 비교
- [ ] 실험 2: 학습률 (1e-4, 5e-5, 1e-5)
- [ ] 실험 3: LoRA target modules
  - attention only (`q_proj, v_proj`)
  - attention + FFN (`q_proj, v_proj, fc1, fc2`)
- [ ] 각 실험 WandB에 기록 + EXPERIMENTS.md에 분석

### Day 4: 한국어 혼합 학습
> AI Hub 승인이 아직 안 났을 경우 → [대안 데이터셋](#대안-한국어-데이터셋) 사용

- [ ] 실험 4: 영어만 vs 한국어만 vs 혼합 (7:3, 5:5, 3:7)
- [ ] 한국어 문서에 대한 성능 별도 측정
- [ ] 최적 데이터 비율 도출

### Day 5: 중간 정리
- [ ] 최적 하이퍼파라미터 조합 확정
- [ ] 최적 모델 체크포인트 Drive/Kaggle에 백업
- [ ] WandB 실험 비교 대시보드 캡처
- [ ] EXPERIMENTS.md 중간 리포트 작성

### 2주차 산출물
- [ ] 파인튜닝된 모델 체크포인트 (Drive 백업 포함)
- [ ] WandB 실험 대시보드 (최소 6~8회 실험)
- [ ] 하이퍼파라미터 ablation 결과표

### 대안 한국어 데이터셋
AI Hub 승인 지연 시 활용:
- CORD v2 (`naver-clova-ix/cord-v2`) 단독으로 계속 진행 (영수증 구조 학습에 충분)
- 직접 스캔한 영수증 (3주차 작업 일부 앞당기기) — 스마트폰으로 촬영 후 라벨링
- `daekeun-ml/naver-news-summarization-ko` (텍스트 포함 문서, 보조 데이터로 활용)

---

## 3주차: 한국어 특화 + 양자화 + 최적화

### Day 1: AI Hub 한국어 데이터 전처리
- [ ] AI Hub 한국어 OCR 데이터 다운로드 (승인 완료 전제)
- [ ] 데이터 포맷 파악: AI Hub는 bounding box + text 형태의 JSON으로 제공됨
- [ ] **Florence-2 입력 포맷으로 변환 코드 작성** (이 작업이 핵심)
  ```
  AI Hub JSON                         Florence-2 타겟 시퀀스
  {                               →   <s_store>GS25</s_store>
    "store": "GS25",                  <s_date>2024.01.01</s_date>
    "date": "2024.01.01",             <s_total>3500</s_total>
    "total": 3500
  }
  ```
- [ ] 변환된 데이터 샘플 10~20개 육안 검수
- [ ] Google Drive / Kaggle Dataset으로 업로드

### Day 2: 데이터 검증 + Augmentation
- [ ] 변환된 데이터셋 PyTorch Dataset에 통합 및 DataLoader 테스트
- [ ] Data Augmentation 적용 (Albumentations)
- [ ] CORD 포맷과 AI Hub 포맷 통합 테스트 (혼합 학습 대비)

### Day 3: 한국어 특화 파인튜닝
- [ ] 2주차 최적 설정 기반으로 한국어 데이터 파인튜닝
- [ ] 한국어 영수증 테스트셋으로 성능 측정
- [ ] 영어 CORD 대비 성능 비교

### Day 4: 양자화 실험 (Kaggle/Colab에서 실행)
- [ ] FP16 변환 + 성능/속도 측정 (거의 무손실, 반드시 적용)
- [ ] INT8 양자화 (bitsandbytes) + 성능/속도 측정
- [ ] INT4 양자화 (QLoRA/bitsandbytes) + 성능/속도 측정
- [ ] 정확도 vs 추론속도 vs 모델크기 비교표 작성
- [ ] 정확도 손실 5% 이내 최적 조합 선택
- **주의**: auto-gptq는 Florence-2 지원이 불안정할 수 있음 → bitsandbytes 우선

### Day 5: 추론 최적화
- [ ] ONNX 변환 시도
  - **주의**: Florence-2는 encoder-decoder + vision encoder 복합 구조라 단순 변환 어려움
  - vision encoder만 별도 ONNX 변환 후 벤치마크 비교
  - 실패 시 TorchScript 또는 `torch.compile()` 대안 시도
- [ ] ~~TensorRT 적용~~ → **Kaggle/Colab 환경에서 TRT 환경 구성 비현실적, 스킵**
  - Kaggle T4에서 TRT 빌드 시간 30분 이상 + 버전 의존성 이슈
  - 대신 `torch.compile(model, mode="reduce-overhead")` 로 대체
- [ ] 최종 추론 속도 비교표 (FP32 vs FP16 vs INT8 vs torch.compile)

### 3주차 산출물
- [ ] 한국어 커스텀 데이터셋 (직접 구축, Drive/Kaggle에 업로드)
- [ ] 양자화 전/후 성능 비교표
- [ ] 최적화된 모델 (추론 속도 목표: FP32 대비 2배+)

---

## 4주차: 서빙 + 데모 + 문서화

### Day 1: FastAPI 서빙
- [ ] POST /predict 엔드포인트 (이미지 업로드 → JSON 응답)
- [ ] 에러 핸들링 (잘못된 이미지, 모델 로드 실패 등)
- [ ] 응답 시간 로깅
- [ ] 로컬에서 uvicorn으로 동작 확인

### Day 2: Gradio 데모
- [ ] 이미지 업로드 → 추출 결과 표시 UI
- [ ] 원본 이미지 위에 인식된 필드 오버레이 시각화
- [ ] 예시 이미지 포함
- [ ] **HuggingFace Spaces에서 Gradio 데모 배포** (무료, GPU 필요 시 ZeroGPU 신청)

### Day 3: Docker + 배포
- [ ] Dockerfile 작성 (멀티스테이지 빌드)
- [ ] docker-compose.yml 작성
- [ ] 로컬 Docker 실행 테스트
- [ ] **모델 가중치 HuggingFace Hub에 업로드** (배포 용이성)

### Day 4: GitHub README
- [ ] 프로젝트 소개 + 아키텍처 다이어그램
- [ ] 성능 표 (베이스라인 vs 파인튜닝 vs 양자화)
- [ ] 데모 GIF 녹화
- [ ] 설치 & 실행 가이드
- [ ] WandB 실험 대시보드 링크
- [ ] Kaggle 노트북 링크 (재현 가능성)

### Day 5: 블로그 + 마무리
- [ ] 기술 블로그 포스트 작성 (실패 경험 포함)
- [ ] 코드 정리 + 주석 보강
- [ ] 최종 테스트

### 4주차 산출물
- [ ] 라이브 데모 URL (HuggingFace Spaces)
- [ ] GitHub 레포 (완성)
- [ ] 기술 블로그 1편
- [ ] HuggingFace Hub 모델 카드

---

## GPU 사용량 예산 계획 (Kaggle 기준)

| 작업 | 예상 시간 | 누적 시간 |
|------|----------|---------|
| 베이스라인 zero-shot | ~0.5h | 0.5h |
| 1차 파인튜닝 (epoch 5) | ~2h | 2.5h |
| rank 실험 x4 (epoch 3) | ~4h | 6.5h |
| lr 실험 x3 (epoch 3) | ~3h | 9.5h |
| target module 실험 x2 | ~2h | 11.5h |
| 데이터 비율 실험 x4 | ~4h | 15.5h |
| 한국어 특화 파인튜닝 | ~2h | 17.5h |
| 양자화 실험 | ~1h | 18.5h |
| ONNX/최적화 | ~1h | 19.5h |
| **합계** | **~20h** | **주당 30h 이내** |

> Colab Pro를 병행하면 실험 횟수 제한 걱정 없이 진행 가능

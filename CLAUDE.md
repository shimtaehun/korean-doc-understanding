# Korean Document Understanding Model

## About This Project
한국어 문서(영수증, 명함, 계약서) 이미지를 입력하면 구조화된 JSON으로 추출하는 멀티모달 모델.
Florence-2-base를 한국어 문서 데이터셋으로 LoRA 파인튜닝하고, 양자화 + FastAPI 서빙까지 완성한다.

## Tech Stack
- **Framework**: PyTorch 2.x
- **Model**: Florence-2-base (microsoft/Florence-2-base-ft)
- **Fine-tuning**: HuggingFace Transformers + PEFT (LoRA)
- **Experiment Tracking**: Weights & Biases (WandB)
- **Quantization**: bitsandbytes (4bit/8bit) — GPTQ는 Florence-2 지원 불안정, bitsandbytes 우선
- **Serving**: FastAPI + Uvicorn
- **Inference Optimization**: ONNX Runtime, torch.compile (TensorRT는 클라우드 환경 한계로 제외)
- **Demo UI**: Gradio (HuggingFace Spaces 배포)
- **Deployment**: Docker, HuggingFace Spaces
- **Training Environment**: Kaggle (T4/P100) 또는 Google Colab — 로컬 GPU 없이 클라우드에서 학습
- **Python**: 3.10+

## Project Structure
```
korean-doc-understanding/
├── CLAUDE.md                    # 이 파일 (Claude Code가 자동으로 읽음)
├── README.md                    # GitHub용 프로젝트 설명
├── docs/
│   ├── ROADMAP.md               # 주차별 로드맵 & 체크리스트
│   ├── ARCHITECTURE.md          # 아키텍처 상세 설계
│   └── EXPERIMENTS.md           # 실험 기록 (결과, 실패 원인 등)
├── configs/
│   ├── train_config.yaml        # 학습 하이퍼파라미터
│   └── serving_config.yaml      # 서빙 설정
├── notebooks/                   # Kaggle/Colab 실행용 노트북
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_train_lora.ipynb
│   ├── 04_hyperparameter_search.ipynb
│   ├── 05_quantization.ipynb
│   └── 06_export_serve.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset 클래스
│   │   ├── preprocessing.py     # 이미지 전처리, augmentation
│   │   └── download.py          # 데이터셋 다운로드 스크립트
│   ├── model/
│   │   ├── florence_lora.py     # LoRA 적용 모델 래퍼
│   │   └── utils.py             # 모델 유틸리티
│   ├── training/
│   │   ├── train.py             # 학습 루프 (PyTorch 직접 작성)
│   │   ├── evaluate.py          # 평가 (F1, accuracy 등)
│   │   └── callbacks.py         # WandB 로깅, 체크포인트 저장
│   ├── optimization/
│   │   ├── quantize.py          # 양자화 실험 (FP16/INT8/INT4)
│   │   └── export_onnx.py       # ONNX 변환 (vision encoder만 지원)
│   └── serving/
│       ├── app.py               # FastAPI 서빙 서버
│       └── gradio_demo.py       # Gradio 데모 UI
├── scripts/                     # 로컬 실행용 (클라우드는 notebooks/ 사용)
│   ├── run_train.sh
│   ├── run_eval.sh
│   └── run_benchmark.sh
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_serving.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Key Commands

### 로컬 (코드 작성 / 단위 테스트)
```bash
# 환경 세팅
pip install -r requirements.txt

# 단위 테스트
pytest tests/ -v

# 서빙 서버 실행 (로컬 추론 테스트용)
uvicorn src.serving.app:app --reload --port 8000

# Docker 빌드 & 실행
docker build -t doc-understanding .
docker run -p 8000:8000 doc-understanding
```

### Kaggle / Google Colab (학습 실행)
```python
# WandB 연동 (노트북 첫 셀)
import os
os.environ["WANDB_API_KEY"] = "your_api_key"  # Kaggle Secrets 또는 Colab userdata 권장

# CORD 데이터셋 로드 (GitHub clone 불필요)
from datasets import load_dataset
dataset = load_dataset("naver-clova-ix/cord-v2")

# Google Drive 마운트 (Colab)
from google.colab import drive
drive.mount('/content/drive')

# 체크포인트 저장 경로
# Kaggle:  /kaggle/working/checkpoints/
# Colab:   /content/drive/MyDrive/korean-doc/checkpoints/

# 학습 실행
!python src/training/train.py --config configs/train_config.yaml

# 평가
!python src/training/evaluate.py --checkpoint /kaggle/working/checkpoints/best_model
```

## Code Standards
- Type hints 필수
- Google style docstring
- 함수는 50줄 이하로 유지
- 모든 실험 결과는 WandB에 기록
- 새 실험 시작 전 docs/EXPERIMENTS.md에 가설 먼저 기록

## Development Workflow
1. 새 기능/실험은 항상 별도 브랜치에서 작업
2. 실험 전 docs/EXPERIMENTS.md에 가설과 예상 결과 기록
3. 실험 후 결과와 분석을 EXPERIMENTS.md에 추가
4. 코드 변경 시 관련 테스트 작성/업데이트
5. 커밋 메시지는 한국어로 작성 (예: "feat: LoRA rank 실험 추가")

## Current Phase
- [ ] 1주차: 환경 세팅 + 데이터 준비 + 베이스라인 → docs/ROADMAP.md 참고
- [ ] 2주차: LoRA 파인튜닝 + 실험
- [ ] 3주차: 한국어 특화 + 양자화
- [ ] 4주차: 서빙 + 데모 + 문서화

## Important Notes
- Florence-2 모델 로드 시 `trust_remote_code=True` 필수
- CORD 데이터셋: `load_dataset("naver-clova-ix/cord-v2")` (HuggingFace 직접 로드)
- 로컬 데이터셋 경로: `data/raw/` (원본), `data/processed/` (전처리 후)
- 체크포인트 경로: Kaggle `/kaggle/working/checkpoints/` / Colab Drive 마운트 후 저장
- WandB 프로젝트명: `korean-doc-understanding`
- GPU 메모리 부족 시 batch_size 줄이고 gradient_accumulation_steps 올리기
- AI Hub 승인은 1~3주 소요 → 프로젝트 시작 즉시 신청할 것
- Label Studio는 로컬에서 실행 (학습/실험만 Kaggle/Colab 사용, 라벨링은 로컬)
- TensorRT는 Kaggle/Colab 환경에서 사용 불가 → torch.compile()로 대체

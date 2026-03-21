# 실험 기록

모든 실험은 아래 포맷으로 기록한다.
WandB run 링크도 함께 첨부할 것.

---

## 실험 템플릿

### 실험 N: [실험 제목]
- **날짜**: YYYY-MM-DD
- **가설**: [이렇게 하면 이런 결과가 나올 것이다]
- **변경 사항**: [베이스라인 대비 뭘 바꿨는지]
- **설정**: [LoRA rank, lr, batch_size 등 주요 파라미터]
- **결과**: [F1, CER, 추론시간 등 수치]
- **분석**: [왜 이런 결과가 나왔는지, 다음에 뭘 해볼지]
- **WandB**: [run 링크]

---

## 베이스라인 (1주차)

### EDA: CORD v2 데이터셋 분석
- **날짜**: 2026-03-17
- **데이터셋**: naver-clova-ix/cord-v2

| 항목 | 결과 |
|------|------|
| Train / Validation / Test | 800 / 100 / 100개 |
| 이미지 너비 | 204 ~ 3024px |
| 이미지 높이 | 364 ~ 4224px |
| 평균 메뉴 아이템 수 | 3.7개 |
| max_length=512 초과 샘플 | 0개 (512 설정으로 전체 커버 가능) |

**필드 분포 (train 800개 기준)**

| 필드 | 출현 횟수 | 비율 |
|------|----------|------|
| menu | 800 | 100% |
| total | 798 | 99.8% |
| sub_total | 548 | 68.5% |
| void_menu | 1 | 0.1% |

**데이터 구조 특이사항 (코드 수정 반영)**
- `menu` 필드가 list가 아닌 dict로 오는 샘플 존재 → 단일 dict를 `[dict]`로 변환 처리
- `sub_total` 필드가 dict가 아닌 list로 오는 샘플 존재 → 첫 번째 dict 요소만 사용

**시사점**
- 이미지 크기 편차가 매우 크므로 768x768 리사이즈 + augmentation 필요
- sub_total 없는 샘플(31.5%)이 많아 optional 필드로 처리 필요 (현재 코드 반영)
- max_length=512로 truncation 없이 전체 학습 가능

### 실험 0: Zero-shot 베이스라인
- **날짜**: 2026-03-17
- **가설**: Florence-2-base는 영어 문서에는 어느 정도 작동하지만 한국어에는 약할 것
- **설정**: 파인튜닝 없음, CORD 테스트셋 20개, 프롬프트 `<DocVQA>`
- **결과**:
  - Field F1: **0.0000**
  - CER: **0.9972**
- **분석**: 파인튜닝 없이는 `<s_menu>`, `<s_total>` 같은 우리 포맷으로 출력하지 못함. F1=0은 예상된 결과. CER≈1.0은 정답과 출력이 완전히 다름을 의미. LoRA 파인튜닝 후 이 수치가 얼마나 올라가는지가 핵심 지표.
- **WandB**: https://wandb.ai/sthun0211-home/korean-doc-understanding/runs/7n66imiy

---

## LoRA 파인튜닝 실험 (2주차)

### 실험 0-1: LoRA 파인튜닝 첫 학습 (파이프라인 검증)
- **날짜**: 2026-03-17
- **가설**: 파인튜닝 후 loss가 내려가고 CER이 개선될 것
- **설정**: rank=8, alpha=16, lr=0.00005, batch_size=2, grad_accum=8 (effective batch=16), epochs=5, target_modules=[q_proj, v_proj]
- **결과**:

  | Epoch | Loss   | Field F1 | CER    |
  |-------|--------|----------|--------|
  | 1     | 4.6253 | 0.0000   | 0.9950 |
  | 2     | 4.0493 | 0.0000   | 0.9966 |
  | 3     | 2.9608 | 0.0000   | 0.9386 |
  | 4     | 1.9945 | 0.0000   | 0.9145 |
  | 5     | 1.3517 | 0.0000   | 0.8673 |

  - Best checkpoint: 6.8 MB (LoRA 어댑터만 저장)

- **분석**:
  - Loss가 4.6 → 1.3으로 꾸준히 감소 → 모델이 정상적으로 학습 중
  - CER이 0.995 → 0.867로 개선 → 텍스트 생성 품질 향상 중
  - Field F1이 0으로 유지 → 아직 `<s_menu>`, `<s_total>` 등 정확한 XML 태그 형식을 생성하지 못함
  - 5 epoch으로는 태그 포맷 학습이 부족한 것으로 보임 → epoch 추가 또는 하이퍼파라미터 탐색 필요
  - 04 노트북(하이퍼파라미터 탐색)에서 최적 설정을 찾은 후 더 많은 epoch으로 재학습 예정

### 실험 1~3: 하이퍼파라미터 탐색 (04 노트북)
- **날짜**: 2026-03-17
- **환경**: Google Colab T4, epoch=3
- **결과**: 모든 실험에서 Field F1 = 0.0000

  | 실험 | 설정 | Field F1 |
  |------|------|----------|
  | rank-4  | r=4,  alpha=8  | 0.0000 |
  | rank-8  | r=8,  alpha=16 | 0.0000 |
  | rank-16 | r=16, alpha=32 | 0.0000 |
  | rank-32 | r=32, alpha=64 | 0.0000 |
  | lr-1e-4 | lr=1e-4 | 0.0000 |
  | lr-5e-5 | lr=5e-5 | 0.0000 |
  | lr-1e-5 | lr=1e-5 | 0.0000 |
  | attn-qv   | q_proj, v_proj | 0.0000 |
  | attn-full | q/k/v/o_proj   | 0.0000 |
  | attn-ffn  | q/v_proj+fc1/2 | 0.0000 |

- **분석**:
  - 3 epoch으로는 XML 태그 포맷(`<s_menu>...</s_menu>`) 학습이 부족해 F1 구분 불가
  - 03 결과(5 epoch, CER 0.867)에서도 동일하게 F1=0 → epoch 부족이 원인
  - F1로 차별화가 불가하므로 **Florence-2 파인튜닝 표준 권장값** 채택:
    - rank=8, alpha=16, lr=5e-5, target_modules=[q_proj, v_proj]
  - **다음 액션**: 위 설정으로 epoch=15로 재학습 → F1 > 0 기대

### 실험 4: LoRA 강화 + 특수 토큰 등록 (30 epoch)
- **날짜**: 2026-03-20
- **환경**: Google Colab T4, epoch=30
- **가설**: rank 확장 + XML 태그 특수 토큰 등록으로 F1 > 0 달성 가능
- **변경 사항**:
  - LoRA r=8→16, alpha=16→32
  - target_modules에 k_proj, out_proj, fc1, fc2 추가
  - learning_rate 5e-5→1e-4
  - CORD 19종 XML 태그를 특수 토큰으로 등록 (임베딩 평균 초기화)
  - embed_tokens/lm_head를 modules_to_save로 full 학습
  - normalize_xml_tags 강화 (s-, s,, s. 등 구분자 변형 처리)
- **결과**:

  | Metric | 값 |
  |--------|-----|
  | Best Field F1 | **0.1856** |
  | val/field_f1 (final) | 0.1855 |
  | val/CER (final) | 0.6387 |
  | train/epoch_loss | 0.0617 |
  | epoch | 30 |
  | global_step | 12,000 |

- **분석**:
  - F1=0 벽을 돌파. XML 태그 포맷 생성 시작
  - CER도 0.87→0.64로 큰 폭 개선
  - train loss가 0.062로 매우 낮음 → 오버피팅 가능성 있음 (val F1이 낮은 이유)
  - F1 0.18은 아직 낮음 → 더 많은 데이터 또는 정규화 강화 필요
  - lr이 30 epoch 끝에서 ~1e-4로 높음 → warmup/cosine 스케줄 점검 필요
- **WandB**: https://wandb.ai/sthun0211-home/korean-doc-understanding/runs/t93kjbex

### 실험 5: 오버피팅 억제 (dropout + lr 조정)
- **날짜**: 2026-03-21
- **환경**: Google Colab T4, epoch=20
- **가설**: train_loss=0.062로 오버피팅이 F1 정체의 주원인. dropout 강화 + lr 낮추면 val F1 개선 가능
- **변경 사항**:
  - lora_dropout: 0.05 → 0.1
  - learning_rate: 1e-4 → 5e-5
  - epochs: 30 → 20
  - warmup_ratio: 0.1 → 0.15
- **설정**: r=16, alpha=32, target_modules=all, weight_decay=0.01
- **결과**:

  | Metric | 값 |
  |--------|-----|
  | Best Field F1 | |
  | val/CER | |
  | train/epoch_loss | |

- **분석**:
- **WandB**:

---

### 실험 4: 데이터 비율 (영어:한국어)
- **날짜**:
- **가설**:
- **설정**:
  | 비율 (영:한) | 영어 F1 | 한국어 F1 | 비고 |
  |-------------|---------|----------|------|
  | 10:0        |         |          |      |
  | 7:3         |         |          |      |
  | 5:5         |         |          |      |
  | 3:7         |         |          |      |
- **분석**:

---

## 양자화 실험 (3주차)

### 실험 5: 양자화 방법 비교
- **날짜**:
- **가설**: INT8까지는 정확도 손실이 미미할 것, INT4부터 주의 필요
- **설정**:
  | 방법  | 모델 크기 | F1    | 추론 시간 (ms) | 정확도 손실 |
  |------|----------|-------|---------------|-----------|
  | FP32 |          |       |               | baseline  |
  | FP16 |          |       |               |           |
  | INT8 |          |       |               |           |
  | INT4 |          |       |               |           |
- **분석**:
- **LG Aimers 경험과 비교**:

---

## 핵심 교훈 (프로젝트 종료 후 정리)

1.
2.
3.

# 노트북 01 — CORD v2 데이터 탐색 (EDA)

> 대응 파일: `notebooks/01_data_exploration.ipynb`

---

## 이 노트북의 목적

모델을 학습시키기 전에 데이터가 어떻게 생겼는지 파악합니다.
데이터를 모르고 코드를 짜면 나중에 버그가 터집니다 (실제로 경험함).

---

## 셀별 설명

### 셀 1 — 레포 클론
```python
!rm -rf /kaggle/working/korean-doc-understanding
!git clone https://github.com/shimtaehun/korean-doc-understanding.git
```
- `!`는 Jupyter에서 shell 명령어를 실행하는 문법
- 매번 실행할 때마다 최신 코드를 받아오기 위해 기존 폴더를 지우고 다시 클론
- Kaggle 세션은 초기화되면 `/kaggle/working/` 내용이 사라지므로 항상 클론 필요

---

### 셀 2 — 경로 추가
```python
import sys
sys.path.append("/kaggle/working/korean-doc-understanding")
```
- Python이 `src/` 폴더를 찾을 수 있도록 경로를 추가
- 이게 없으면 `from src.data.dataset import ...`할 때 `ModuleNotFoundError` 발생

---

### 셀 3 — 패키지 설치
```python
!pip install -q datasets transformers matplotlib seaborn pandas
```
- Kaggle에 이미 설치된 패키지도 있지만 버전 문제로 명시적으로 설치
- `-q` 옵션: quiet mode (설치 로그 최소화)

---

### 셀 4 — CORD 데이터셋 로드
```python
dataset = load_dataset("naver-clova-ix/cord-v2")
```

**`load_dataset()`이 하는 일**
- HuggingFace Hub에서 데이터셋을 다운로드
- 자동으로 train/validation/test split으로 나눠줌
- 로컬에 캐시해서 두 번째 실행부터는 빠름

**CORD v2 구조**
```
dataset["train"][0] =
{
    "image": PIL.Image (영수증 사진),
    "ground_truth": '{"gt_parse": {"menu": [...], "total": {...}}}' # JSON 문자열
}
```
- `ground_truth`는 JSON 문자열이라서 `json.loads()`로 파싱 필요

---

### 셀 5 — 이미지 크기 분포
```python
widths, heights = [], []
for sample in dataset["train"]:
    w, h = sample["image"].size
    widths.append(w)
    heights.append(h)
```

**왜 확인하냐?**
- Florence-2는 입력 이미지를 768×768로 고정
- 원본 이미지 비율이 다양하면 리사이즈 시 왜곡 발생
- 극단적으로 작거나 큰 이미지가 있으면 전처리 전략을 다르게 가져가야 함

**실제 결과**
- W: 204 ~ 3024px, H: 364 ~ 4224px → 편차가 매우 큼
- 리사이즈 필수, padding 추가도 고려 가능

---

### 셀 6 — 필드 분포 분석
```python
from collections import Counter
field_counter = Counter()
for sample in dataset["train"]:
    gt = json.loads(sample["ground_truth"])["gt_parse"]
    for key in gt.keys():
        field_counter[key] += 1
```

**Counter란?**
- 딕셔너리처럼 사용하는 집계 도구
- `counter["menu"] += 1`을 반복하면 각 키가 몇 번 등장했는지 집계

**실제 결과**
| 필드 | 출현 횟수 | 의미 |
|------|----------|------|
| menu | 800 | 항상 존재 |
| total | 798 | 거의 항상 존재 |
| sub_total | 548 | 없는 영수증도 많음 |
| void_menu | 1 | 무효 항목, 무시해도 됨 |

→ `sub_total`은 없는 경우가 많으니 optional로 처리해야 한다는 걸 알 수 있음

---

### 셀 7 — 타겟 시퀀스 변환 검증
```python
from src.data.dataset import cord_to_target_sequence

for i in range(5):
    sample = dataset["train"][i]
    gt = json.loads(sample["ground_truth"])
    target_seq = cord_to_target_sequence(gt)
    print(target_seq[:300])
```

**왜 검증하냐?**
- `cord_to_target_sequence()`가 올바르게 동작하는지 눈으로 확인
- 실제 결과가 이렇게 나오면 정상:
```
<s_menu><s_menuitem><s_nm>Nasi Campur</s_nm><s_price>75,000</s_price></s_menuitem></s_menu>
<s_total><s_total_price>75,000</s_total_price></s_total>
```

---

### 셀 8 — max_length 초과 검사
```python
seq_lengths = []
for sample in dataset["train"]:
    seq = cord_to_target_sequence(json.loads(sample["ground_truth"]))
    seq_lengths.append(len(seq.split()))
```

**왜 중요하냐?**
- `train_config.yaml`에 `max_length: 512`로 설정
- 512를 넘는 샘플은 잘림(truncation) 발생 → 정답이 잘려서 학습 품질 저하
- 결과: 초과 샘플 0개 → 512 설정으로 충분

---

## 이 노트북에서 배울 것

1. HuggingFace `load_dataset()` 사용법
2. PIL Image 다루기 (`.size`, `.convert()`)
3. `json.loads()`로 JSON 문자열 파싱
4. `Counter`로 빈도 집계
5. Matplotlib으로 분포 시각화

---

## 실행 결과 요약 (실제 수치)

| 항목 | 수치 |
|------|------|
| Train / Val / Test | 800 / 100 / 100 |
| 이미지 크기 범위 | W 204~3024, H 364~4224 |
| 평균 메뉴 아이템 수 | 3.7개 |
| max_length=512 초과 | 0개 |

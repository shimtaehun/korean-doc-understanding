# Kaggle 환경 트러블슈팅 기록

> 01~03 노트북 실행 중 발생한 오류와 해결법 정리.
> 새 노트북 작성 시 `notebooks/kaggle_setup.py`를 참고해 공통 셋업 적용할 것.

---

## 오류 1: ModuleNotFoundError - src 모듈 없음

**오류 메시지**
```
ModuleNotFoundError: No module named 'src'
```

**원인**
- 노트북만 Kaggle에 올렸을 때 `src/` 폴더가 없음
- `sys.path`에 레포 경로가 추가되지 않음

**해결**
```python
!git clone https://github.com/shimtaehun/korean-doc-understanding.git
import sys
sys.path.append("/kaggle/working/korean-doc-understanding")
```

---

## 오류 2: IndentationError

**오류 메시지**
```
IndentationError: unexpected indent
```

**원인**
- 코드 복사 시 앞에 공백이 딸려오는 현상
- `!git clone`과 `import sys`를 같은 셀에 넣으면 발생

**해결**
- 셀을 분리 (`!` 명령어 셀과 Python 코드 셀은 반드시 분리)
- 직접 타이핑하거나 공백 제거

---

## 오류 3: BackendError - Kaggle Secrets 없음

**오류 메시지**
```
BackendError: No user secrets exist for kernel id ... and label WANDB_API_KEY
```

**원인**
- 새 노트북을 열 때마다 Secrets 접근 허용 토글을 켜야 함
- 토글이 꺼진 상태에서 실행

**해결**
- **Add-ons → Secrets → WANDB_API_KEY 토글 ON** (노트북마다 매번)

---

## 오류 4: AttributeError - additional_special_tokens

**오류 메시지**
```
AttributeError: TokenizersBackend has no attribute additional_special_tokens
```

**원인**
- 최신 `transformers`와 Florence-2 processor 코드 간 호환 문제
- `tokenizers` 라이브러리 버전 불일치

**해결**
```python
!pip install -q "transformers==4.47.0" "tokenizers==0.21.0"
```

---

## 오류 5: ImportError - EncoderDecoderCache

**오류 메시지**
```
ImportError: cannot import name 'EncoderDecoderCache' from 'transformers'
```

**원인**
- Kaggle 기본 `peft==0.18.1`이 `transformers>=4.45.0`의 `EncoderDecoderCache`를 요구
- `transformers==4.41.0`에는 해당 클래스 없음

**해결**
- `peft`를 다운그레이드하지 말고 `transformers`를 올리기
```python
!pip install -q "transformers==4.47.0" "tokenizers==0.21.0"
```

---

## 오류 6: ImportError - flash_attn 없음

**오류 메시지**
```
ImportError: This modeling file requires the following packages: flash_attn
```

**원인**
- Florence-2 모델 코드가 `flash_attn` import를 시도
- Kaggle T4/P100에 `flash_attn` 미설치 (설치 시 컴파일 30분+)

**해결**
```python
import sys, types, importlib.machinery
import transformers.utils.import_utils

for mod_name in ['flash_attn', 'flash_attn.flash_attn_interface', 'flash_attn.bert_padding']:
    mock = types.ModuleType(mod_name)
    mock.__spec__ = importlib.machinery.ModuleSpec(mod_name, loader=None)
    sys.modules[mod_name] = mock

transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
```
모델 로드 시 `attn_implementation="eager"` 옵션 추가 (이미 `florence_lora.py`에 반영됨)

---

## 오류 7: ValueError - flash_attn.__spec__ is None

**오류 메시지**
```
ValueError: flash_attn.__spec__ is None
```

**원인**
- `types.ModuleType`으로 만든 mock 모듈의 `__spec__`이 None
- `importlib.util.find_spec()`이 None spec을 만나면 ValueError 발생

**해결**
- mock 모듈에 `ModuleSpec` 명시적 추가 (오류 6 해결코드에 반영됨)

---

## 오류 8: RuntimeError - torchvision::nms does not exist

**오류 메시지**
```
RuntimeError: operator torchvision::nms does not exist
```

**원인**
- `pip install --target=/kaggle/working/packages`로 설치 시 `torch`까지 해당 경로에 설치됨
- 시스템 `torchvision`이 `/kaggle/working/packages/torch`를 참조하면서 충돌

**해결**
- `--target` 방식 사용 금지
- 잔여 폴더 정리 후 재시작
```python
!rm -rf /kaggle/working/packages
# Session → Restart Session
```

---

## 오류 9: TypeError - CORD v2 menu 구조 불일치

**오류 메시지**
```
TypeError: string indices must be integers, not 'str'
```

**원인**
- CORD v2 일부 샘플에서 `menu` 필드가 list가 아닌 dict로 옴

**해결** (`src/data/dataset.py`에 반영됨)
```python
if isinstance(menu_items, dict):
    menu_items = [menu_items]
for item in menu_items:
    if not isinstance(item, dict):
        continue
```

---

## 오류 10: AttributeError - sub_total/total 구조 불일치

**오류 메시지**
```
AttributeError: 'list' object has no attribute 'items'
```

**원인**
- CORD v2 일부 샘플에서 `sub_total`, `total` 필드가 dict가 아닌 list로 옴

**해결** (`src/data/dataset.py`에 반영됨)
```python
if isinstance(sub_total, list):
    sub_total = sub_total[0] if sub_total and isinstance(sub_total[0], dict) else {}
if isinstance(total, list):
    total = total[0] if total and isinstance(total[0], dict) else {}
```

---

## 오류 11: ModuleNotFoundError - jiwer 없음

**오류 메시지**
```
ModuleNotFoundError: No module named 'jiwer'
```

**원인**
- `evaluate.py`와 `02_baseline.ipynb`에서 CER 계산에 `jiwer` 사용
- 기존 pip install 셀에 `jiwer`가 누락됨

**해결**
```python
!pip install -q "transformers==4.47.0" "tokenizers==0.21.0" "jiwer"
```
- **현재 모든 노트북(02~04)의 셀 3에 반영됨**

---

## 오류 12: TypeError - learning_rate str

**오류 메시지**
```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

**원인**
- YAML에서 `5e-5`를 PyYAML이 문자열로 파싱
- `AdamW(lr=cfg["training"]["learning_rate"])` 에서 str이 전달됨

**해결**
- `train_config.yaml`에서 `learning_rate: 5e-5` → `learning_rate: 0.00005`로 변경 (반영됨)
- 노트북에서 즉시 해결: `lr=float(cfg["training"]["learning_rate"])`

---

## 오류 13: RuntimeError - Input type (float) and bias type (c10::Half)

**오류 메시지**
```
RuntimeError: Input type (float) and bias type (c10::Half) should be the same
```

**원인**
- 학습은 `autocast(float16)` 컨텍스트 안에서 실행되어 모델 가중치가 float16 상태
- `evaluate()` 함수가 `autocast` 없이 `model.generate()` 호출 → input float32 vs 가중치 float16 충돌

**해결** (`src/training/evaluate.py`에 반영됨)
```python
with autocast("cuda", dtype=torch.float16):
    output_ids = model.generate(...)
```

---

## 공통 셋업 체크리스트

새 노트북 실행 전 매번 확인:
- [ ] **Add-ons → Secrets → WANDB_API_KEY 토글 ON**
- [ ] 셀 1: `!rm -rf /kaggle/working/packages` + `!rm -rf /kaggle/working/korean-doc-understanding` + `!git clone`
- [ ] 셀 2: `sys.path.append(...)` (Python 코드 셀 분리)
- [ ] 셀 3: `pip install "transformers==4.47.0" "tokenizers==0.21.0" "jiwer"`
- [ ] 셀 4: flash_attn mock
- [ ] 설치 후 **Session → Restart Session** (패키지 버전 충돌 시)

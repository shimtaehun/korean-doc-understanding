"""Kaggle 공통 셋업 — 모든 노트북의 셀 4에서 실행."""

import importlib.machinery
import sys
import types

import transformers.utils.import_utils

# flash_attn mock (Kaggle T4/P100에 미설치)
for mod_name in ["flash_attn", "flash_attn.flash_attn_interface", "flash_attn.bert_padding"]:
    mock = types.ModuleType(mod_name)
    mock.__spec__ = importlib.machinery.ModuleSpec(mod_name, loader=None)
    sys.modules[mod_name] = mock

transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
print("flash_attn mock 완료")

"""학습 콜백 — WandB 로깅, 체크포인트 저장."""

from pathlib import Path
from typing import Optional

import torch
import wandb


class WandBCallback:
    """WandB 로깅 콜백.

    Args:
        log_interval: 몇 step마다 로깅할지
    """

    def __init__(self, log_interval: int = 10) -> None:
        self.log_interval = log_interval
        self._global_step = 0

    def on_step_end(self, loss: float, lr: float, epoch: int) -> None:
        self._global_step += 1
        if self._global_step % self.log_interval == 0:
            wandb.log({
                "train/loss": loss,
                "train/lr": lr,
                "train/global_step": self._global_step,
                "epoch": epoch,
            })

    def on_epoch_end(self, epoch: int, train_loss: float, val_metrics: dict) -> None:
        wandb.log({
            "train/epoch_loss": train_loss,
            "val/field_f1": val_metrics.get("field_f1", 0.0),
            "val/cer": val_metrics.get("cer", 1.0),
            "epoch": epoch,
        })

    def on_train_end(self, best_metric: float) -> None:
        wandb.summary["best_field_f1"] = best_metric
        wandb.finish()


class CheckpointCallback:
    """체크포인트 저장 콜백.

    Args:
        checkpoint_dir: 체크포인트 저장 루트 경로
        save_best_only: True면 best model만 저장
        metric_name: 기준 메트릭 이름
        higher_is_better: 높을수록 좋은 메트릭이면 True
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_best_only: bool = True,
        metric_name: str = "field_f1",
        higher_is_better: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self._best_metric = float("-inf") if higher_is_better else float("inf")

    def is_better(self, current: float) -> bool:
        if self.higher_is_better:
            return current > self._best_metric
        return current < self._best_metric

    def on_epoch_end(
        self,
        model,
        processor,
        epoch: int,
        val_metrics: dict,
    ) -> bool:
        """체크포인트 저장. best 갱신 시 True 반환.

        Args:
            model: 저장할 모델
            processor: 저장할 processor
            epoch: 현재 에폭
            val_metrics: 검증 메트릭 딕셔너리

        Returns:
            best 갱신 여부
        """
        current = val_metrics.get(self.metric_name, 0.0)
        is_best = self.is_better(current)

        if not self.save_best_only:
            self._save(model, processor, f"epoch_{epoch}")

        if is_best:
            self._best_metric = current
            self._save(model, processor, "best_model")
            print(f"  ✓ Best model 저장 (epoch={epoch}, {self.metric_name}={current:.4f})")

        return is_best

    def _save(self, model, processor, dirname: str) -> None:
        save_dir = self.checkpoint_dir / dirname
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        processor.save_pretrained(str(save_dir))

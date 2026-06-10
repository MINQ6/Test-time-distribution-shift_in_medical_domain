from __future__ import annotations

import json
import os
from typing import Any

from utils.training.common import logger


class WandbLogger:
    def __init__(self):
        self._wandb = None
        self.enabled = False

    def init(
        self,
        *,
        enabled: bool,
        project: str | None,
        run_name: str | None,
        entity: str | None,
        config: dict[str, Any],
    ) -> None:
        if not enabled:
            self.enabled = False
            return

        try:
            import wandb  # type: ignore
        except ModuleNotFoundError:
            logger.warning("wandb is not installed. Continuing without wandb logging.")
            self.enabled = False
            return

        if not hasattr(wandb, "init"):
            logger.warning(
                "wandb import succeeded but does not expose init(). "
                "This usually means the wandb package is not installed and a local 'wandb/' directory "
                "was imported instead. Continuing without wandb logging."
            )
            self.enabled = False
            return

        self._wandb = wandb
        try:
            self._wandb.init(
                project=project,
                name=run_name,
                entity=entity,
                config=config,
            )
        except Exception as exc:
            logger.warning("wandb initialization failed: %s. Continuing without wandb logging.", exc)
            self._wandb = None
            self.enabled = False
            return
        self.enabled = True
        logger.info("wandb run initialized: project=%s name=%s", project, run_name)

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.log(payload, step=step)

    def summary_update(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self._wandb is None:
            return
        for key, value in payload.items():
            self._wandb.summary[key] = value

    def log_analysis_artifacts(self, artifact_paths: dict[str, str]) -> None:
        if not self.enabled or self._wandb is None:
            return

        image_payload = {}
        for key in ("svd_heatmap", "svd_dendrogram", "qr_heatmap", "qr_dendrogram"):
            path = artifact_paths.get(key)
            if path and os.path.exists(path):
                image_payload[f"analysis/{key}"] = self._wandb.Image(path)
        if image_payload:
            self._wandb.log(image_payload)

        for key in ("svd_summary", "qr_summary"):
            path = artifact_paths.get(key)
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as file_obj:
                    summary = json.load(file_obj)
                prefix = f"analysis/{key.replace('_summary', '')}"
                self.summary_update(
                    {
                        f"{prefix}/mean_off_diagonal_similarity": summary["mean_off_diagonal_similarity"],
                        f"{prefix}/min_off_diagonal_similarity": summary["min_off_diagonal_similarity"],
                        f"{prefix}/max_off_diagonal_similarity": summary["max_off_diagonal_similarity"],
                        f"{prefix}/closest_to_zero_pair": str(summary["closest_to_zero_pair"]),
                        f"{prefix}/highest_similarity_pair": str(summary["highest_similarity_pair"]),
                    }
                )

    def finish(self) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.finish()
        self.enabled = False


wandb_logger = WandbLogger()

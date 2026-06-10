"""4-GPU 병렬 client 학습 (FedDPA-F).

한 라운드 안에서 client들은 서로 독립이므로, client를 GPU에 1:1 배정해 병렬 학습한다.
- spawn 프로세스로 격리, 각 child가 CUDA_VISIBLE_DEVICES를 첫 CUDA 호출 전에 설정
- 입출력은 모두 CPU state dict(picklable)
- num_gpus < num_clients이면 wave 단위로 나눠 처리

opt-in: feddpa_f가 PARALLEL_CLIENT_GPUS>1일 때만 사용. 기본은 기존 순차 경로.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import torch
import torch.multiprocessing as mp

from utils.training.common import checkpoint_suffix, logger
from utils.training.wandb import wandb_logger


def resolve_num_gpus(default: int = 1) -> int:
    """PARALLEL_CLIENT_GPUS env로 병렬 GPU 수 결정. 1이면 병렬 비활성."""
    raw = os.getenv("PARALLEL_CLIENT_GPUS")
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _worker(device_id: str, kind: str, kwargs: Dict[str, Any], result_path: str, error_path: str) -> None:
    """child 프로세스: 지정 GPU에서 client 1개를 학습하고 결과를 파일로 저장.

    device_id는 SLURM이 실제 할당한 GPU id(부모 CUDA_VISIBLE_DEVICES의 한 항목).
    텐서를 Queue로 보내면 torch가 공유메모리 fd 방식을 써서 child 종료 시 깨진다.
    따라서 결과는 torch.save로 디스크에 쓰고, 부모가 읽는다.
    """
    # 첫 CUDA 호출 전에 GPU 고정 (CUDA_VISIBLE_DEVICES는 CUDA init 시점에 읽힘)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    src_root = str(Path(__file__).resolve().parents[2])
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    try:
        import torch as _torch
        from utils.training.local_round import (
            run_local_federated_round,
            run_local_personalization_round,
        )

        if kind == "round":
            result = run_local_federated_round(**kwargs)
        elif kind == "personalization":
            result = run_local_personalization_round(**kwargs)
        else:
            raise ValueError(f"Unknown kind: {kind}")
        _torch.save(result, result_path)
    except Exception:  # noqa: BLE001
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())


def _run_tasks_parallel(
    tasks: List[Tuple[str, str, Dict[str, Any]]],
    num_gpus: int,
) -> Dict[str, Any]:
    """tasks: [(client_id, kind, kwargs), ...] -> {client_id: result}. GPU당 1개씩 wave 처리.

    결과는 child가 디스크에 저장하고 부모가 join 후 읽는다 (공유메모리 fd 문제 회피).
    """
    ctx = mp.get_context("spawn")
    results: Dict[str, Any] = {}
    # SLURM이 실제 할당한 GPU 목록 (0-base가 아닐 수 있음, 부분 사용 노드 대응)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_list = [g.strip() for g in visible.split(",") if g.strip()] or [str(i) for i in range(num_gpus)]
    logger.info("[Parallel] allocated GPUs (CUDA_VISIBLE_DEVICES)=%s -> using %s", visible or "<unset>", gpu_list)
    tmpdir = tempfile.mkdtemp(prefix="feddpa_parallel_")
    saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        for wave_start in range(0, len(tasks), num_gpus):
            wave = tasks[wave_start:wave_start + num_gpus]
            procs = []
            meta: List[Tuple[str, str, str]] = []
            for local_idx, (client_id, kind, kwargs) in enumerate(wave):
                device_id = gpu_list[local_idx % len(gpu_list)]
                rp = os.path.join(tmpdir, f"{client_id}.pt")
                ep = os.path.join(tmpdir, f"{client_id}.err")
                # 부모가 start() 전에 env 설정 → child가 처음부터 단일 GPU로 태어남
                # (child가 import 후 env를 바꾸면 "CUDA_VISIBLE_DEVICES changed after start" 에러)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
                p = ctx.Process(target=_worker, args=(device_id, kind, kwargs, rp, ep))
                p.start()
                procs.append(p)
                meta.append((client_id, rp, ep))
                logger.info("[Parallel] launched %s on GPU %s (wave %d)", client_id, device_id, wave_start // num_gpus)

            for p in procs:
                p.join()

            for (client_id, rp, ep), p in zip(meta, procs):
                if os.path.exists(ep):
                    raise RuntimeError(f"[Parallel] client {client_id} failed:\n{open(ep).read()}")
                if not os.path.exists(rp):
                    raise RuntimeError(
                        f"[Parallel] client {client_id} produced no result "
                        f"(process exitcode={p.exitcode}; likely crashed/OOM)."
                    )
                results[client_id] = torch.load(rp, map_location="cpu")
                logger.info("[Parallel] collected %s (exitcode=%s)", client_id, p.exitcode)
    finally:
        # 부모 env 원복 (자식 spawn 위해 잠시 바꿨던 CUDA_VISIBLE_DEVICES)
        if saved_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd
        shutil.rmtree(tmpdir, ignore_errors=True)
    return results


# =========================================================================
# Stage 1 / warm-up: collect_client_uploads 병렬 버전
# =========================================================================

def collect_client_uploads_parallel(
    server,
    round_idx: int,
    *,
    epochs_override: int | None,
    phase_label: str,
    num_gpus: int,
    stage1_only: bool = True,
    train_mode: str = "feddpa",
) -> Dict[str, Dict]:
    effective_epochs = epochs_override or server.local_epochs
    logger.info(
        "[Parallel Phase 1] %s | round=%d epochs=%d | %d GPUs",
        phase_label, round_idx, effective_epochs, num_gpus,
    )

    tasks: List[Tuple[str, str, Dict[str, Any]]] = []
    for client_id, client in server.clients.items():
        if client.local_lora_state is None or client.global_lora_state is None:
            raise ValueError(f"{client_id} has no initial local/global LoRA state. Broadcast state first.")
        kwargs = dict(
            model_id=server.model_id,
            client_id=client_id,
            dataset_name=server.dataset_name,
            round_idx=round_idx,
            local_state=client.get_adapter_state(),
            global_lora_state=client.global_lora_state,
            epochs=effective_epochs,
            micro_batch_size=server.micro_batch_size,
            gradient_accumulation_steps=server.gradient_accumulation_steps,
            num_samples=server.num_samples,
            max_length=server.max_length,
            seed=server.seed,
            rank=server.rank,
            lora_alpha=server.lora_alpha,
            lora_dropout=server.lora_dropout,
            learning_rate=server.learning_rate,
            train_on_inputs=server.train_on_inputs,
            lora_target_modules=server.lora_target_modules,
            stage1_only=stage1_only,
            train_mode=train_mode,
            mixer_state=getattr(client, "mixer_state", None),
        )
        tasks.append((client_id, "round", kwargs))

    results = _run_tasks_parallel(tasks, num_gpus)

    uploads: Dict[str, Dict] = {}
    for client_id, client in server.clients.items():
        result = results[client_id]
        # 부모 프로세스에서 client state 갱신 (순차판의 run_client_round와 동일)
        client.set_adapter_state(result["local_state"])
        if result.get("mixer_state") is not None:
            client.mixer_state = result["mixer_state"]
        uploads[client_id] = result["upload_payload"]
        uploads[client_id]["metrics"] = result.get("metrics", {})
        metrics = result.get("metrics", {})
        if result["upload_payload"].get("had_non_finite"):
            logger.warning("[Parallel Phase 1] %s produced non-finite global LoRA values (round %d).", client_id, round_idx)
        wandb_logger.log(
            {
                "train/round": round_idx,
                f"train/{client_id}/global_final_loss": metrics.get("global_final_loss"),
                f"train/{client_id}/global_epoch_losses": metrics.get("global_epoch_losses"),
                f"train/{client_id}/global_had_non_finite": result["upload_payload"].get("had_non_finite"),
            },
            step=round_idx,
        )
        logger.info("[Parallel Phase 1] Collected upload from %s", client_id)
    return uploads


# =========================================================================
# Stage 2: local personalization 병렬 버전
# =========================================================================

def run_stage2_local_personalization_parallel(
    server,
    checkpoints_dir: str,
    *,
    num_gpus: int,
) -> dict[str, str]:
    from utils.training.federated import _save_personalized_checkpoints

    logger.info("[Parallel Stage 2] Local personalization | %d GPUs", num_gpus)
    tasks: List[Tuple[str, str, Dict[str, Any]]] = []
    for client_id, client in server.clients.items():
        if client.global_lora_state is None:
            raise ValueError(f"{client_id} is missing assigned global LoRA for personalization.")
        kwargs = dict(
            model_id=server.model_id,
            client_id=client_id,
            dataset_name=server.dataset_name,
            aligned_global_state=client.global_lora_state,
            epochs=server.personalization_epochs,
            micro_batch_size=server.micro_batch_size,
            gradient_accumulation_steps=server.gradient_accumulation_steps,
            num_samples=server.num_samples,
            max_length=server.max_length,
            seed=server.seed + 10_000 + int(checkpoint_suffix(client_id)),
            rank=server.rank,
            lora_alpha=server.lora_alpha,
            lora_dropout=server.lora_dropout,
            learning_rate=server.learning_rate,
            train_on_inputs=server.train_on_inputs,
            lora_target_modules=server.lora_target_modules,
        )
        tasks.append((client_id, "personalization", kwargs))

    results = _run_tasks_parallel(tasks, num_gpus)

    metrics_by_client: dict[str, float | None] = {}
    for client_id, client in server.clients.items():
        result = results[client_id]
        client.local_lora_state = result["local_state"]
        metrics_by_client[client_id] = result["metrics"]["personalization_final_loss"]
        wandb_logger.log(
            {
                f"personalization/{client_id}/final_loss": result["metrics"]["personalization_final_loss"],
                f"personalization/{client_id}/epoch_losses": result["metrics"]["personalization_epoch_losses"],
            },
            step=server.num_rounds,
        )

    saved_paths = _save_personalized_checkpoints(server, checkpoints_dir)
    mean_loss = mean([v for v in metrics_by_client.values() if v is not None]) if metrics_by_client else None
    wandb_logger.summary_update({"personalization/mean_final_loss": mean_loss})
    return saved_paths

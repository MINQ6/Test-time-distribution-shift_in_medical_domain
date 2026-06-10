"""Local-finetuned baseline (FedDPA paper Table 1).

Each client trains independently on its own task data — NO federation.
For fair comparison with FL methods, total compute = num_rounds × local_epochs.
Result is saved per-client in the same format as FedDPA-F checkpoints, so the
existing `inference_fedDPA.py` can evaluate it with `--*_adapter_mode global_only`
(the trained weights are placed in the dual-LoRA `global` slot by
`run_local_federated_round`, the `local` slot is untouched at init).
"""
import os

import torch

from utils.training.common import logger
from utils.training.local_round import run_local_federated_round


def _checkpoint_dir() -> str:
    ts = os.environ.get("RUN_TIMESTAMP")
    if not ts:
        raise RuntimeError("RUN_TIMESTAMP is not set; refusing to write checkpoints to a non-TS path")
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "checkpoints", ts))


def run_local_finetuned_training(server) -> dict[str, str]:
    """Train each client locally on its own data for the same total compute as FL."""
    checkpoints_dir = _checkpoint_dir()
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Match FL total compute: num_rounds * local_epochs effective epochs per client
    total_epochs = server.local_epochs * server.num_rounds
    logger.info(
        "[Local-finetuned] starting | clients=%d | epochs/client=%d (= %d rounds * %d local_epochs)",
        len(server.client_ids), total_epochs, server.num_rounds, server.local_epochs,
    )

    saved_paths: dict[str, str] = {}
    for client_id in server.client_ids:
        logger.info("[Local-finetuned] === %s === training %d epochs on own data", client_id, total_epochs)
        result = run_local_federated_round(
            model_id=server.model_id,
            client_id=client_id,
            dataset_name=server.dataset_name,
            round_idx=1,
            local_state=None,
            global_lora_state=None,
            epochs=total_epochs,
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
            stage1_only=True,
        )
        save_path = f"{checkpoints_dir}/dual_lora_adapter_{client_id}.pth"
        torch.save(result["local_state"], save_path)
        saved_paths[client_id] = save_path
        logger.info("[Local-finetuned] saved %s -> %s", client_id, save_path)

    logger.info("[Local-finetuned] done | %d clients saved under %s", len(saved_paths), checkpoints_dir)
    return saved_paths


def train_local_client(
    *,
    model_id: str,
    client_id: str,
    dataset_name: str,
    epochs: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    num_samples: int,
    max_length: int,
    seed: int,
    rank: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
    lora_target_modules: list[str],
    train_on_inputs: bool = False,
    checkpoint_prefix: str = "local_finetuned_dual_lora",
    checkpoint_dir: str | None = None,
) -> str:
    """Single-client local fine-tuning helper. Kept for back-compat with
    `utils.train` re-exports. Trains one client's dual LoRA in stage-1-only
    mode (global slot only) and saves it under the given prefix."""
    result = run_local_federated_round(
        model_id=model_id,
        client_id=client_id,
        dataset_name=dataset_name,
        round_idx=1,
        local_state=None,
        global_lora_state=None,
        epochs=epochs,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_samples=num_samples,
        max_length=max_length,
        seed=seed,
        rank=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        learning_rate=learning_rate,
        train_on_inputs=train_on_inputs,
        lora_target_modules=lora_target_modules,
        stage1_only=True,
    )
    if checkpoint_dir is None:
        checkpoint_dir = _checkpoint_dir()
    os.makedirs(checkpoint_dir, exist_ok=True)
    suffix = client_id.replace("client_", "")
    save_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_client_{suffix}.pth")
    torch.save(result["local_state"], save_path)
    logger.info("[Local-finetuned] %s saved -> %s", client_id, save_path)
    return save_path
